# rwkv_contrib

A collection of RWKV related Python modules I use in my projects.

The project setup is extremely weird and homegrown, so it is advised to not use this as a dependency, but rather read the source code and copy the parts you need.

## BNF-like grammar

A BNF-like grammar was implemented to constrain the output of the model. It is not a full BNF implementation, but rather a subset of it. (And at least a bit more intuitive than the full BNF.)

Please refer to `bnf.py`, `bnf_complex.py`, `bnf_simple` for the implementation. For grammar defined, please check `easy_schema.py` and `json_schema.py`. For an actual implemented pipeline, check `grammar_pipeline.py`.

## Debugging tools

The `debug_tools.py` contains a few debugging tools I use in my projects. It is mainly used to dump the state of the model.
