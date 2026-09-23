"""The miner walks its own order, proves what it generated, and resyncs on refusal."""

from types import SimpleNamespace

from reliquary.corpus.encoding import prompt_token_ids
from reliquary.corpus.walk import walk_index
from reliquary.miner.corpus_miner import Generation, build_submission, mine_steps

EOS = 99


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)


def _job(prompt_count=50, n=2):
    return SimpleNamespace(job_id="math-v1", prompt_count=prompt_count, eos_token_id=EOS,
                           checkpoint_sha256="a" * 64, sampling=SimpleNamespace(n=n),
                           prompt_order="miner_walk")


class _Generator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt_ids, n):
        self.prompts.append(prompt_ids)
        return [Generation(tokens=[104, 105, EOS], proofs=["AAAA"]) for _ in range(n)]


class _Client:
    def __init__(self, answers):
        self.answers = list(answers)
        self.submitted = []
        self.cursor_reads = 0
        self.position = 0

    def cursor(self, hotkey):
        self.cursor_reads += 1
        return self.position

    def submit(self, body):
        self.submitted.append(body)
        answer = self.answers.pop(0)
        if answer == "accepted":
            self.position += 1
        return {"reason": answer, "accepted": answer == "accepted"}


def test_the_submission_carries_the_walk_prompt_and_its_text():
    body = build_submission(job=_job(), hotkey="5Hot", cursor=0, prompt_index=7, rendered_prompt="q7",
                            generations=[Generation([104, 105, EOS], ["AAAA"])],
                            tokenizer=_Tokenizer(), sign=lambda b: "sig")
    assert body["prompt_index"] == 7 and body["signature"] == "sig"
    assert body["completions"] == [{"tokens": [104, 105, EOS], "text": "hi", "proofs": ["AAAA"]}]


def test_the_miner_follows_its_own_walk():
    client, generator = _Client(["accepted"] * 3), _Generator()
    mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator, tokenizer=_Tokenizer(),
               render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=3)
    assert [b["prompt_index"] for b in client.submitted] == [walk_index("math-v1", "5Hot", c, 50) for c in range(3)]
    assert [b["cursor"] for b in client.submitted] == [0, 1, 2]


def test_a_refused_step_resynchronises_the_cursor():
    client = _Client(["prompt_full", "accepted"])
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=2)
    assert counts == {"prompt_full": 1, "accepted": 1}
    assert client.cursor_reads >= 2


def test_a_complete_job_stops_the_miner():
    client = _Client(["job_complete", "accepted"])
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=5)
    assert counts == {"job_complete": 1} and len(client.submitted) == 1


def test_miner_and_auditor_tokenize_the_prompt_identically():
    generator = _Generator()
    mine_steps(job=_job(), hotkey="5Hot", client=_Client(["accepted"]), generator=generator,
               tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=1)
    index = walk_index("math-v1", "5Hot", 0, 50)
    assert generator.prompts == [prompt_token_ids(_Tokenizer(), f"q{index}")]
