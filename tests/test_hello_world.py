import pytest


from hello_world import add

# I want to import from hello_world.py I import the module ../ out and test so that I keep organized the files

def test_adding():

    sum = add(2, 3)

    assert sum == 5