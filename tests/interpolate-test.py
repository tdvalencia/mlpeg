'''
    Tests model interpolation
'''

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//compress')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..//decompress')))

from compress import decimate
from decompress import upscale

