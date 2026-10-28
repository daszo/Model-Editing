"""DOME: additive DocID model editing for generative retrieval.

Everything DOME does is shared with MOSE and lives in `common`. DOME has no
logic of its own, so this file holds only what is unique to it: which editor
class to use and which weight update to apply.

`common` is an extraction of the original DOME.py. The pristine upstream file
is preserved in git and can be restored at any time with:

    git checkout 778cb53 -- DOME.py
"""

from typing import Any, Dict, List

from common import ModelAndTokenizer, T5DocIDDOME, run


def apply_dome(
        editor: T5DocIDDOME,
        mt: ModelAndTokenizer,
        requests: List[Dict],
        args: Any,
) -> ModelAndTokenizer:
    """Stage 7 for DOME: the constrained closed-form additive update."""
    return editor.apply_dome_edit(mt=mt, edit_targets=requests)


if __name__ == "__main__":
    run(
        editor_cls=T5DocIDDOME,
        apply_edit=apply_dome,
        banner="T5-DOME: DocID Editing for Generative Retrieval",
        done_banner="DOME EDITING COMPLETE",
    )
