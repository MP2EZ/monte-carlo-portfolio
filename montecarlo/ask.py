"""
Describe a run in one sentence (roadmap N8).
===========================================

    python3 run.py --ask "165 against 167, with the return-model check"

Claude Haiku turns the sentence into a run spec — the same JSON `--spec` takes,
`--again` replays and `persistence` records — and the runner shows it for a yes
before anything runs. The model can only PROPOSE: every proposal goes through
`run.parse_spec` and `glide`'s own validation, so a typo'd portfolio, a
malformed schedule or an invented bond sleeve is refused exactly as a typed
spec would be. A sentence the spec cannot express comes back as `unsupported`
rather than as a nearby run that answers a different question.

The schema's enums are built from the registries the runner validates against
(policies, guardrail responses, plan choices, bond sleeves, portfolio ids), so
the two cannot disagree.

Needs the `anthropic` package and credentials (ANTHROPIC_API_KEY, or an
`ant auth login` profile). Nothing else in the model imports this module.
"""

import json

from montecarlo import glide
from montecarlo import policies as _policies
from montecarlo.loaders import BOND_SLEEVES

ASK_MODEL = 'claude-haiku-5-5'


def _nullable_enum(values):
    return {'type': ['string', 'null'], 'enum': sorted(values) + [None]}


def _schema():
    import run as runner
    return {
        'type': 'object',
        'properties': {
            'mode': {'type': 'string', 'enum': ['retirement', 'accumulation']},
            'portfolios': {'type': 'array', 'items': {'type': 'string'}},
            'custom': {'anyOf': [{'type': 'array', 'items': {'type': 'string'}},
                                 {'type': 'null'}]},
            'policy': _nullable_enum(_policies.POLICIES),
            'gk_response': _nullable_enum(_policies.GK_RESPONSE),
            'return_views': {'type': ['boolean', 'null']},
            'compare_choice': {'anyOf': [
                {'type': 'object',
                 'properties': {
                     'dimension': {'type': 'string', 'enum': sorted(runner.PLAN_CHOICES)},
                     'values': {'type': 'array', 'items': {'type': 'number'}}},
                 'required': ['dimension', 'values'],
                 'additionalProperties': False},
                {'type': 'null'}]},
            'verify_sims': {'type': ['integer', 'null']},
            'unsupported': {'type': ['string', 'null']},
        },
        'required': ['mode', 'portfolios', 'custom', 'policy', 'gk_response',
                     'return_views', 'compare_choice', 'verify_sims', 'unsupported'],
        'additionalProperties': False,
    }


SPEC_SCHEMA = _schema()


def _system(portfolios):
    import run as runner
    catalogue = '\n'.join(f"  {pid}: {p.get('name', '')}" for pid, p in portfolios.items())
    choices = ', '.join(sorted(runner.PLAN_CHOICES))
    return f"""You turn one sentence into a run spec for a retirement Monte Carlo model.
Fill every field. Use null for anything the sentence does not ask for — the
runner supplies defaults. Never invent a portfolio id, sleeve or field.

PORTFOLIOS (the only valid ids):
{catalogue}

FIELDS
- mode: "retirement" (spending in retirement; the default) or "accumulation"
  (wealth to the plan horizon, no withdrawals).
- portfolios: the ids being compared. Always at least one.
- custom: only when the sentence asks for a bond sleeve or a change of
  portfolio at some age. A list of glide specs; when used, it is the complete
  list of strategies to run, so include the plain portfolios too. Syntax:
    "141"                    hold 141 throughout
    "141>141+10@65"          from age 65 hold 141 with 10% bonds
    "141>141+10@65>141@70"   bonds from 65, back to 141 at 70
    "166>166+5:stip_only@60>166@65"  a 5% sleeve of one bond type, 60 to 65
  Bond sleeves after ":": {', '.join(sorted(BOND_SLEEVES))}. Omit ":..." for
  the portfolio's usual mix. Ages strictly increase.
- policy: the withdrawal rule, if named.
- gk_response: how fast guyton_klinger follows a rising portfolio, if named.
- return_views: true when the sentence asks whether the result holds under
  other return models / if the return model is wrong / robustness.
- compare_choice: when the sentence compares a decision for ONE portfolio:
  dimension one of {choices}; values the options as numbers (ages, or a
  savings multiplier such as 0.5, 1.0, 1.5).
- verify_sims: a path count, only if the sentence gives one.
- unsupported: if the sentence asks for something these fields cannot express
  (a market assumption sweep, a new fund, a date), say what in one sentence and
  leave the rest as close to empty as possible. Otherwise null."""


def propose(sentence, portfolios, client=None):
    """Ask the model for a proposal: a dict matching SPEC_SCHEMA. Not yet a spec."""
    if client is None:
        try:
            import anthropic
        except ImportError:
            raise ValueError("--ask needs the anthropic package: "
                             "pip install anthropic") from None
        client = anthropic.Anthropic()
    response = client.messages.create(
        model=ASK_MODEL,
        max_tokens=4096,
        system=_system(portfolios),
        messages=[{'role': 'user', 'content': sentence}],
        output_config={'format': {'type': 'json_schema', 'schema': SPEC_SCHEMA},
                       'effort': 'low'},
    )
    if response.stop_reason == 'refusal':
        raise ValueError("the model declined to propose a spec for that sentence")
    if response.stop_reason == 'max_tokens':
        raise ValueError("the proposal was cut off before it was complete")
    text = next((b.text for b in response.content if b.type == 'text'), None)
    if text is None:
        raise ValueError("the model returned no spec")
    return json.loads(text)


def to_spec(proposal, portfolios):
    """A proposal accepted the way a typed --spec is, or ValueError."""
    import run as runner
    if proposal.get('unsupported'):
        raise ValueError(f"the spec cannot express that: {proposal['unsupported']}")
    spec = {k: v for k, v in proposal.items()
            if v is not None and k != 'unsupported'}
    spec = runner.parse_spec(json.dumps(spec), portfolios)
    age = runner.default_config()['age']
    for gspec in spec.get('custom') or []:
        glide.validate(glide.parse_schedule(gspec, age), portfolios)
    return spec


def spec_from_sentence(sentence, portfolios, client=None):
    return to_spec(propose(sentence, portfolios, client=client), portfolios)
