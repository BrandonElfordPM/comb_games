#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from gym.envs.registration import register

register(id='DiskonnectPlayerEnv-v0',
         entry_point='Diskonnect1D.envs:DiskonnectPlayerEnv')
