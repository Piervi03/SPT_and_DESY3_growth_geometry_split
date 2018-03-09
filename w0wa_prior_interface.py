def setup(options):
    pass

def execute(block):
    w0 = block.get_double('cosmological_parameters', 'w')
    wa = block.get_double('cosmological_parameters', 'wa')
    if (w0+wa) < 0:
        return 0
    else:
        print "w0+wa=%.3f"%(w0+wa)
        return 1

def cleanup(config):
    pass
