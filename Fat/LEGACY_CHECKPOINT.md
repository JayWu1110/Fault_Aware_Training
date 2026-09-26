# `sample_best.ckpt`

This file is a weight dump from the **original incomplete script**
(analog pixels at train time, no real fault injection in the forward pass).
Tests use it only to check that the FC state-dict keys still match.

It is **not** a FAT model from this codebase. Paper / Table I numbers
must come from `outputs/paper/` (or whatever `--out-dir` you pass to
`run_table1.py`), never from this file.
