;;;; cl-nn.asd

(asdf:defsystem #:cl-nn
  :serial t
  :description "A neural network implementation in Common Lisp."
  :author "Nicolas Avrutin <nicolasavru@gmail.com>"
  :depends-on (#:alexandria)
  :components ((:file "package")
               (:file "cl-nn")))

