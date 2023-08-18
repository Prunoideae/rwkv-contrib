from os import environ
import warnings
from bnf_complex import RepeatNode, SequenceNode
from bnf_simple import NotCharNode
from easy_schema import literal

from pipeline import GenerationArgs

# suppress warnings in v5 model
warnings.filterwarnings("ignore", category=UserWarning)
if True:
    # setup RWKV_JIT_ON and RWKV_CUDA_ON
    environ["RWKV_CUDA_ON"] = "1"
    environ["RWKV_JIT_ON"] = "1"
    from rwkv.model import RWKV

from rwkv_contrib.bnf import BNFTree, setup_tokens
from rwkv_contrib.tokenizer import RWKVTokenizer
from rwkv_contrib import bnf
from rwkv_contrib.grammar_pipeline import BNFPipeline

summary_tree = BNFTree()

# setup tokenizers for bnf to use
# as the implementation will filter available logits
bnf.base_tokenizer = RWKVTokenizer().tokenizer
setup_tokens()

# state is frozen in the pipeline to reduce overhead for prompt.
firestarter = ":"
summary_prompt = f"""
Instruction: Summarize the input email in 1-2 sentences, and suggest what to do next.

Input: 
Title: Let’s start your Green Diet! <catering@ust.hk>
Email: Follow Conference Lodge on Facebook and Instagram to keep posted of the latest
news and promotions
\-------------------------------------------------------
Please do not reply to this email message.
\-------------------------------------------------------
To unsubscribe from this communication, please visit webpage:
https://myaccount.ust.hk/refreshable_lists

Response
""".strip()

# define the grammar
# basically, it matches a following pattern:
#  This email is about (...). It says (...).
# So, as a wise assistant, I think you should (...).
# <EOS>
summary_matcher = SequenceNode(
    summary_tree,
    [
        literal(summary_tree, b" This email is about "),
        RepeatNode(summary_tree, NotCharNode(summary_tree, b",.!?\n")),
        literal(summary_tree, b". It says "),
        RepeatNode(summary_tree, NotCharNode(summary_tree, b".,\n")),
        literal(summary_tree, b".\nSo, as a wise assistant, I think you should "),
        RepeatNode(summary_tree, NotCharNode(summary_tree, b".!?\n")),
        literal(summary_tree, b".\n"),
    ],
)


# create rwkv instance, look at the temperature and top_p here
# also don't forget to set a correct model path
rwkv = RWKV(model="../models/RWKV-4-World-7B-v1-20230626-ctx4096.pth", strategy="cuda fp16")
args = GenerationArgs(
    temperature=2.5,
    top_p=0.6,
    alpha_frequency=0.3,
    alpha_presence=0.3,
)

# create the pipeline
# a model can be used for multiple pipelines
# the logits_cache is used to cache the allowed logits for each node
summary_pipeline = BNFPipeline(
    rwkv,
    summary_tree,
    summary_matcher,
    logits_cache="logits_cache_summary.npz",
    default_args=args,
)

# infer the state from the prompt
# so for a continuing generation, the state can be reused for multiple times
print(summary_prompt + firestarter, end="")
_, state = summary_pipeline.infer(summary_pipeline.encode(summary_prompt))

# actually generate the summary, note that the state is modified in-place
# if multiple generations are needed, the state should be deepcopied
for partial in summary_pipeline.generate(firestarter, 256, state=state):
    print(partial, end="", flush=True)

# dump the logits cache, by default it will be saved to the cache in the pipeline
summary_pipeline.dump_logits()
