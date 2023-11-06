from hoomd.md.pair import pair

class FEPerturbationPair(pair):
    def __init__(self, charge_value=float('inf'), type_override='', **kwargs):
        super().__init__(**kwargs)
        self._param_dict.update(
            ParameterDict(charge_value=float(charge_value)))
        self._param_dict.update(
            ParameterDict(type_override=str(type_override)))

