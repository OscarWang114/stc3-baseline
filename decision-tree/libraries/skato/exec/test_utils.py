#!/usr/bin/env python
# -*- coding: utf-8 -*-


from skato.exec.utils import main


def test_utils(**kwargs):
    print(kwargs)
    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    main(__file__, globals())
