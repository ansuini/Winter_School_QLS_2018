# Winter School QSB 2018

Didactic materials for the :

Winter School on Quantitative Systems Biology: Learning and Artificial Intelligence | (smr 3246)

http://indico.ictp.it/event/8339/


Contents:

- Installation instructions (below)
- References to books, papers and other relevant writings
- Jupyter notebooks


### Basic rules of accessing help

These are a few common sense basic rules to solve problems and ask for help :

- if it doesn't work look into the documentation 
  - man pages (Linux)
  - python documentation (ex. on a Jupyter cell evaluate ?function/module to access help, or ??function/module to access code)
  - from the Internet (online documentation, forums etc.)
  
- if this either doesn't work look for help from colleagues

- if this either doesn't work call one of us

- if this either doesn't work ... (ok this should not happen so often !)

In any case the most important rule is :

- do not get stack more than a few minutes (say 5-10 depending on the problem) in trying to solve things by yourself !


### Installation instructions


All the code will be developed in Python 3.6.

- Download Anaconda Python 3.6 https://www.anaconda.com/download/
- Install Git (https://git-scm.com/) on your laptop 

  - Linux/macOS : $sudo apt-get install git
  - Windows 10 : install Ubuntu from Store (https://tutorials.ubuntu.com/tutorial/tutorial-ubuntu-on-windows#0) 
  - Windows (earlier versions) : install Cygwin (http://www.cygwin.com/) and then install git from the available packages
  
  Alternative for Windows users : install VirtualBox (https://www.virtualbox.org/wiki/Downloads) and then follow instructions
  for Linux/macOS users
  
- Create (if you do not have already) your github account https://github.com/
- Fork repository REPO NAME [link]
- Clone repository REPO NAME [link]
- Install OpenAI Gym (https://gym.openai.com/)
  - on a terminal : $ pip install gym


### GPU access

Training is not particularly instructive in a first approach, where ideas are important.
So, when a long training is required (more than a few minutes), we could shift to 
demonstrative sessions with just one computer, and its monitor made visible by a projector.

Then we can make the code, the trained models and/or the results accessible for 
download and experimentation. What do you think ?

