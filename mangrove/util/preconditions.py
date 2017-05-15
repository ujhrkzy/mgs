# -*- coding: utf-8 -*-
#
# This file is a part of software developed by Unirobot Inc.
# Copyright 2016 Unirobot Inc. All Rights Reserved.
# The source code in this file is the property of Unirobot Inc.,
# and may not be copied, distributed, modified or sold except under a
# licence expressly granted by Unirobot Inc. to do so.
# ==============================================================================


def check_not_null(value):
    if value is None:
        raise ValueError("Value is null.")
    return value

