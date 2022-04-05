"""
TODO
"""

from otanalysis.controller import Controller
import pandas as pd

class TestController:
    """
    TODO
    """
    @classmethod
    def setup_class(cls):
        directory_test = 'software_test/data_test/verif'
        cls.controller = Controller(None, directory_test)


    # test si la list de fichier est bien crée
    def test_create_list_files(self):
        """
        TODO
        """
        assert len(self.controller.files) > 0

    def test_length_dict_curve(self):
        """
        TODO
        """
        assert len(self.controller.dict_curve) > 0

    #test si un fichier incomplet e reconnu incomplet
    def test_incomplete_file(self):
        #self.controller.files = []
        print(self.controller.files)
        file_incomplete = 'software_test/data_test/verif/b5c5-2021.06.07-15.10.03.254.jpk-nt-force'
        new_curve, check_incomplete_file = Controller.create_object_curve(file_incomplete, 30, 50)
        assert new_curve == None and check_incomplete_file == True

    def test_output(self, tmpdir):
        repository_output = tmpdir.mkdir('Result')
        name_file = self.controller.output_save(["--dest", str(repository_output)])
        with open(name_file, 'r') as file_test:
            assert file_test.readline()

    @classmethod
    def teardown_class(cls):
        if cls.controller:
            del cls.controller
    

    
