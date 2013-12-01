;;;; cl-nn.lisp

(in-package #:cl-nn)

;;; "cl-nn" goes here. Hacks and glory await!

(defstruct neuron
  ;; A vector input weights. The ith weight corresponds to the ith
  ;; neuron in the previous layer.
  in-weights
  bias-weight
  fixed-input
  ;; activation function
  g
  ;; derivative of activation function
  dg
  ;; weighted sum of of inputs
  (in 1.0 :type float)
  ;; output; a = g(in)
  (a 1.0 :type float)
  ;; delta for back-propogation
  (delta 0.0 :type float))

(defun sigmoid (x)
  (/ 1 (1+ (exp (- x)))))

(defun dsigmoid (x)
  (let ((y (sigmoid x)))
    (* (- 1 y) y)))

(defun make-network (sizes &key (g #'sigmoid) (dg #'dsigmoid) (fixed-input -1.0) (weights))
  "Respresent network as a vector of lists, where each list represents a layer."
  (let ((network (make-array (length sizes) :element-type 'list :initial-element nil))
        (pos 0))
    (do* ((prev-size nil (car sizes))
          (sizes sizes (cdr sizes))
          (size (car sizes) (car sizes))
          (layer () ()))
         ((null size))
      (dotimes (i size)
        (let ((in-weights (make-array prev-size :element-type 'float :initial-element 0.0))
              (bias-weight))
          (if prev-size
              (progn
                (setf bias-weight (pop weights))
                (dotimes (n prev-size)
                  (setf (aref in-weights n)
                        (if weights (pop weights) (random 0.1))))))
          (push (make-neuron :in-weights in-weights :g g :dg dg :in fixed-input
                             :fixed-input fixed-input :bias-weight bias-weight)
                layer)))
      (setf (aref network pos) layer)
      (incf pos))
    network))

(defun load-network (filename)
  (let ((in (open filename))
        (num-inputs)
        (num-hidden)
        (num-outputs)
        (line)
        (weights))
    (setf line
          (read-from-string (concatenate 'string
                                         "("
                                         (string-trim '(#\Return) (read-line in))
                                         ")")))
    (setf num-inputs (first line))
    (setf num-hidden (second line))
    (setf num-outputs (third line))

    (dotimes (n num-hidden)
      (setf line
            (read-from-string (concatenate 'string
                                           "("
                                           (string-trim '(#\Return) (read-line in))
                                           ")")))
      (setf weights (append weights line)))

    (dotimes (n num-outputs)
      (setf line
            (read-from-string (concatenate 'string
                                           "("
                                           (string-trim '(#\Return) (read-line in))
                                           ")")))
      (setf weights (append weights line)))

    (make-network (list num-inputs num-hidden num-outputs) :weights weights)))

(defun load-data (filename)
  (let ((in (open filename))
        (num-examples)
        (num-inputs)
        (num-outputs)
        (line)
        (examples)
        (inputs)
        (outputs))
    (setf line
          (read-from-string (concatenate 'string
                                         "("
                                         (string-trim '(#\Return) (read-line in))
                                         ")")))
    (setf num-examples (first line))
    (setf num-inputs (second line))
    (setf num-outputs (third line))

    (dotimes (n num-examples)
      (setf line
            (read-from-string (concatenate 'string
                                           "("
                                           (string-trim '(#\Return) (read-line in))
                                           ")")))
      (setf inputs (subseq line 0 num-inputs))
      (setf outputs (subseq line num-inputs))
      (push (list inputs outputs) examples))
    examples))


(defun back-prop-learning (training-data network &key (alpha 0.1))
  "Train NETWORK on TRAINING-DATA using back-propogation. Training
  data is in the form ((inputs) (outputs))."
  (do ((num-layers (length network))
       (epoch 0 (1+ epoch)))
      ((= epoch 100) network)
    (dolist (datum training-data)
      ;; set input layer outputs
      (map nil #'(lambda (neuron x)
                   (setf (neuron-in neuron) x))
           (aref network 0) (car datum))
      ;; propogate inputs forward
      (map-iota #'(lambda (n)
                    (let ((layer (elt network n))
                          (prev-layer (elt network (1- n)))
                          (i 0))
                      (dolist (neuron layer)
                        (setf (neuron-in neuron)
                              (* (neuron-fixed-input neuron)
                                 (neuron-bias-weight neuron)))
                        (dolist (input-neuron prev-layer)
                          (incf (neuron-in neuron)
                                (* (elt (neuron-in-weights neuron) i)
                                   (neuron-a input-neuron))))
                        (setf (neuron-a neuron)
                              (funcall (neuron-g neuron) (neuron-in neuron))))))
                (1- num-layers) :start 1)
      ;; set output layer deltas
      (map nil #'(lambda (neuron y)
                   (setf (neuron-delta neuron)
                         (* (funcall (neuron-dg neuron) (neuron-in neuron))
                            (- y (neuron-a neuron)))))
           (aref network (1- num-layers)) (cadr datum))
      ;; propogate deltas backward
      (map-iota #'(lambda (n)
                    (let ((layer (elt network n))
                          (next-layer (elt network (1+ n)))
                          (i 0))
                      (dolist (neuron layer)
                        ;; compute delta
                        (dolist (output-neuron next-layer)
                          (incf (neuron-delta neuron)
                                (* (elt (neuron-in-weights output-neuron) i)
                                   (neuron-delta output-neuron)))
                          (incf i))
                        (setf (neuron-delta neuron)
                              (* (funcall (neuron-dg neuron) (neuron-in neuron))
                                 (neuron-delta neuron)))
                        ;; update weights to next layer from current neuron
                        (incf (neuron-bias-weight neuron)
                              (* alpha
                                 (neuron-fixed-input neuron)
                                 (neuron-delta neuron)))
                        (dolist (output-neuron next-layer)
                          (setf (neuron-in-weights output-neuron)
                                (map '(vector float)
                                     #'(lambda (w neuron)
                                         (+ w (* alpha
                                                 (neuron-a neuron)
                                                 (neuron-delta output-neuron))))
                                     (neuron-in-weights neuron) layer))))))
                (- num-layers 2) :start (- num-layers 2) :step -1))))
