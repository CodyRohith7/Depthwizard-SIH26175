"""DepthWizard -- SIH26175 single-view height estimation and 3D fly-through.

This package is organized per the Phase 0 audit's approved architecture
(io -> depth -> geometry -> scale -> uncertainty -> height -> reconstruction
-> evaluation, plus backend/frontend). As of Milestone M1, only `io` and
`depth` contain real logic; the remaining modules are structural stubs for
future milestones -- see each subpackage's docstring for its status.
"""

__version__ = "0.1.0"
