# -*- coding: utf-8 -*-
"""Testing for Analysis Filter"""


from dbdc.common import Dialogue


from unittest import TestCase
import nose
from nose.tools import eq_


class DialogueTestCase(TestCase):

    def setUp(self):
        pass

    def test_parse_annotations(self):
        annotations = [{"breakdown": "O"} for _ in range(16)] \
            + [{"breakdown": "T"} for _ in range(4)] \
            + [{"breakdown": "X"} for _ in range(10)]
        c, ave, var = Dialogue.parse_annotations(annotations)

        eq_(c["O"], 16)
        eq_(c["T"], 4)
        eq_(c["X"], 10)
        eq_(ave, 6.0 / 30)
        eq_(var, 24.8 / 30)

        annotations = [{"breakdown": "O"} for _ in range(30)]
        c, ave, var = Dialogue.parse_annotations(annotations)

        eq_(c["O"], 30)
        eq_(c["T"], 0)
        eq_(c["X"], 0)
        eq_(ave, 1.0)
        eq_(var, 0.0)

    def tearDown(self):
        print("done")
