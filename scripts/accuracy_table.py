#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys

with open(sys.argv[1], 'r') as f:
    d = json.load(f)

d = {x: y.get('last020', y.get('last20')) for x, y in d.items()}
l = list(d.items())
for x in sorted([('/'.join(x[0].split('/')[-6:-2]), x[1]) for x in l]):
    print('%.2f %s' % (x[1], x[0]))
