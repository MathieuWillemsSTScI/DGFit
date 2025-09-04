from dgfit.dustmodel import MRN77DustModel


def test_mrn_initialize():
    dmod = MRN77DustModel()
    assert dmod.sizedisttype == "MRN77"
