"""
Utilities for instrumenting a torch model.

Reconstruction of ROME's util/nethook.py (Bau et al.). DOME.py uses:
    * nethook.TraceDict(module, layers, retain_input=, retain_output=, edit_output=)
      -> a dict-like context manager where tr[layer].input / tr[layer].output hold
         the captured activations, and edit_output can rewrite a layer's output.
    * nethook.get_parameter(model, name)  -> named Parameter
    * nethook.set_requires_grad(flag, *models)

Trace works by registering forward hooks that (optionally) retain the input and
output of the named submodules, optionally editing the output on the fly.
"""
import contextlib
import copy
import inspect
from collections import OrderedDict

import torch


class Trace(contextlib.AbstractContextManager):
    """
    To retain the output of the named layer during the computation of the given
    network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and its output
    will be retained. By default, a direct reference to the output object is
    returned, but options can control this:

        clone=True   - retain a copy of the output, which can be used later
                       even if the model is changed in-place afterwards.
        detach=True  - retain a detached reference or copy.
        retain_grad=True - request gradient to be retained on the output.
        retain_input=True - also retain the input.
        retain_output=False - can disable retaining the output.
        edit_output=fn - calls the function to modify the output before
                         retaining/returning it. fn(output, name) -> new output.
        stop=True    - throws a StopForward exception after the layer is run,
                       to save time computing later layers.
    """

    def __init__(
        self,
        module,
        layer=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        retainer = self
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)

        def retain_hook(m, inputs, output):
            if retain_input:
                retainer.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  # retain_grad applies to output only.
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation. This helps in a case where the
                # output is used in a way that the gradient wouldn't
                # otherwise be retained (e.g. in-place operations).
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation of the
    given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    (output, layer) and returns a modified output; it is applied at every
    traced layer.
    """

    def __init__(
        self,
        module,
        layers=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()


class StopForward(Exception):
    """
    If the only output needed from running a network is the retained submodule
    then Trace(submodule, stop=True) will stop execution immediately after the
    retained submodule by raising the StopForward() exception which is caught by
    the Trace object during the retention phase.
    """

    pass


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is true,
    the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    elif isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        return x


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all params of the given model(s).
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def invoke_with_optional_args(fn, *args, **kwargs):
    """
    Invokes a function with only the arguments that it declares, ignoring other
    arguments. This allows edit_output functions to declare any subset of
    (output, layer), positionally or by keyword.
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name if possible, else in order.
    for i, name in enumerate(argspec.args):
        if name in kwargs:
            pass_args.append(kwargs[name])
            used_kw.add(name)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # Fill unmatched positional args from remaining keyword args in order.
    if unmatched_pos:
        for name, value in kwargs.items():
            if name in used_kw or name in argspec.kwonlyargs:
                continue
            if not unmatched_pos:
                break
            pass_args[unmatched_pos[0]] = value
            del unmatched_pos[0]
            used_kw.add(name)
    # Pass remaining kw args if they are accepted.
    pass_kw = {
        name: value
        for name, value in kwargs.items()
        if name not in used_kw and (name in argspec.kwonlyargs or argspec.varkw is not None)
    }
    # Pass remaining positional args if they are accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)
