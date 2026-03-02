"""Loss functions for multi-instance retrieval."""

from ekmr.losses.adaptive_mimm import AdaptiveMIMMoss
from ekmr.losses.mil_nce import MILNCELoss
from ekmr.losses.sms_loss import SMSLoss

__all__ = ["SMSLoss", "MILNCELoss", "AdaptiveMIMMoss"]
