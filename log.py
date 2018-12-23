'''
How we log.
'''

import logging

# ------------------------------------------------------------------------------

# Keep it simple for now
LOG = logging

LOG.basicConfig(
    format='[%(asctime)s %(threadName)s %(filename)s:%(lineno)d %(levelname)s] %(message)s',
    level=logging.INFO
)
