"""
TODO
"""
from pathlib import Path
from controller.controller import Controller

class TestController:
    """
    TODO
    """
    
    directory_test = Path('software_test/data_test/verif')
    controller = Controller(None, directory_test)

    def test_length_dict_curve(self):
        """
        TODO
        """
        assert len(self.controller.dict_curve) > 0

    

    
