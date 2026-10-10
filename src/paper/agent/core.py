from onmep.agent import (
    BaseAgent as _BaseAgent,
    EIG as _EIG,
    EVR as _EVR,
    TTD as _TTD,
    PEST as _PEST,
    FocusedEIG as _FocusedEIG,
)

from paper.agent.generative import GenerativeMixin
from paper.agent.likelihood import LikelihoodMixin


class BaseAgent(GenerativeMixin, LikelihoodMixin, _BaseAgent):
    ...


class EIG(GenerativeMixin, LikelihoodMixin, _EIG):
    ...
    

class EVR(GenerativeMixin, LikelihoodMixin, _EVR):
    ...


class TTD(GenerativeMixin, LikelihoodMixin, _TTD):
    ...


class PEST(GenerativeMixin, LikelihoodMixin, _PEST):
    ...


class FocusedEIG(GenerativeMixin, LikelihoodMixin, _FocusedEIG):
    ...
