# cl-nn

cl-nn is a neural network implementation in Common Lisp. Training is
done with backpropagation.

## Usage

### Installation

1. Install quicklisp. $ represents a bash shell and * an Lisp REPL.
```$ wget http://beta.quicklisp.org/quicklisp.lisp && sbcl --load quicklisp.lisp
   * (quicklisp-quickstart:install)
   * (ql:add-to-init-file)
   * (sb-ext:exit)
```


2. Configure ASDF2:
```$ mkdir -p ~/.config/common-lisp/source-registry.conf.d/```

~/.config/common-lisp/source-registry.conf.d/projects.conf should
contain the following contents:
```(:tree (:home "Documents/lisp/"))```

"Documents/lisp/" is a path relative to your home directory that ASDF
will search for lisp projects in.

3. Run cl-nn:
```$ sbcl --load run.lisp```

For the sake of your sanity, suggest running sbcl through rlwrap:
```$ rlwrap sbcl```

### Runtime

Upon startup, you will be prompted to train or run a neural
network. If you choose to train one, you will be prompted to generate
a new network (and specify the appropriate layer sizes) or load a
neural network from a file. You will then be prompted for a trainding
files, learning rate, and a number of epochs to run for.

If you choose to run a neural network, you will be prompted for a file
containg a (presumably trained) neural network, a test data file, and
an output file.
