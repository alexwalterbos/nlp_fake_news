from setuptools import setup

setup(
    name='nlp_fake_news',
    version=0.1,
    description='Partial reproduction of Cisco-Talos\' winning Fake News Challenge solution',
    url='https://github.com/alexwalterbos/nlp_fake_news',
    author='Alex Walterbos, Michiel van der Berg & Tom Brunner',
    author_email='atw.231@gmail.com',
    zip_safe=False,
    install_requires=[
        # The scipy stack
        'numpy',
        'scipy',
        'pandas',
        # The natural language toolkit
        'nltk',
        # Expansion for pickle
        'dill',
		
	#vector space modelling toolkit
	'gensim',
	#machine learning lib with a lot of usefull tools
	'sklearn',
    #The tree itself
    'xgboost' #note installation of xgboost may prove difficult: https://stackoverflow.com/questions/36519898/windows-xgboost-error
    ]
)