# -*- coding: utf-8 -*-
"""Testing for Analysis Filter"""


from skato.nlp.utils import JaTokenizer


from unittest import TestCase
import nose
from nose.tools import eq_


class JaTokenizerTestCase(TestCase):

    def setUp(self):
        self.tokenizer = JaTokenizer()
        pass

    def test_tokenize(self):
        u = "私はメカブと申します。"

        tokens = self.tokenizer.tokenize(u)

        eq_(tokens[4]["token"], "申し")
        eq_(tokens[4]["stemmed"], "申す")
        eq_(tokens[5]["pos"], "助動詞")

    def tearDown(self):
        print("done")
