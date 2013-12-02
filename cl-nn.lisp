;;;; cl-nn.lisp

(in-package #:cl-nn)

;;; "cl-nn" goes here. Hacks and glory await!

(setf *READ-DEFAULT-FLOAT-FORMAT* 'double-float)

(defstruct neuron
  ;; A vector input weights. The ith weight corresponds to the ith
  ;; neuron in the previous layer.
  in-weights
  (bias-weight 1.0d0 :type double-float)
  (fixed-input 1.0d0 :type double-float)
  g                                  ; activation function
  dg                                 ; derivative of activation function
  (in 1.0d0 :type double-float)        ; weighted sum of of inputs
  (a 1.0d0 :type double-float)         ; output; a = g(in)
  (delta 0.0d0 :type double-float))    ; delta for backpropogation

(defun sigmoid (x)
  (/ 1 (1+ (exp (- x)))))

(defun dsigmoid (x)
  (let ((y (sigmoid x)))
    (* (- 1 y) y)))

(defun make-network (sizes &key (g #'sigmoid) (dg #'dsigmoid) (fixed-input -1.0d0) (weights))
  "Create a new neural network. Respresent the network as a list of
   lists of neurons, where each list represents a layer."
  (let ((network))
    (do* ((prev-size nil (car sizes))
          (sizes sizes (cdr sizes))
          (size (car sizes) (car sizes))
          (layer () ()))
         ((null size))
      (dotimes (i size)
        (let ((in-weights (make-array prev-size :element-type 'double-float :initial-element 0.0d0))
              (bias-weight (- (random 1.0d0) 0.5)))
          (if prev-size
              (progn
                (if weights (setf bias-weight (pop weights)))
                (dotimes (n prev-size)
                  (setf (elt in-weights n)
                        (if weights (pop weights) (- (random 1.0d0) 0.5))))))
          (push (make-neuron :in-weights in-weights :g g :dg dg :in fixed-input
                             :fixed-input fixed-input :bias-weight bias-weight)
                layer)))
      (push (reverse layer) network))
    (reverse network)))

(defun string-to-list (str)
  "Convert a string of whitespace-separated values to a list."
  (read-from-string
   (concatenate 'string "(" (string-trim '(#\Return) str) ")")))

(defun load-network (filename)
  (let ((in (open filename))
        (num-inputs)
        (num-hidden)
        (num-outputs)
        (line)
        (weights))
    (setf line (string-to-list (read-line in)))
    (setf num-inputs (first line)
          num-hidden (second line)
          num-outputs (third line))

    (dotimes (n (+ num-hidden num-outputs))
      (setf line (string-to-list (read-line in)))
      (setf weights (append weights line)))

    (make-network (list num-inputs num-hidden num-outputs) :weights weights)))

(defun write-network (network filename)
  (with-open-file (out filename :direction :output
                                :if-exists :supersede
                                :if-does-not-exist :create)
    (let ((sizes (mapcar #'length network))
          (weights (mapcar #'(lambda (layer)
                               (mapcar #'(lambda (neuron)
                                           (cons (neuron-bias-weight neuron)
                                                 (coerce (neuron-in-weights neuron) 'list)))
                                       layer))
                           (cdr network))))       ; skip the input layer
      (format out "~{~d~^ ~}~%" sizes)
      (format out "~{~{~{~,3f~^ ~}~%~}~}" weights))))

(defun load-data (filename)
  (let ((in (open filename))
        (num-examples)
        (num-inputs)
        (num-outputs)
        (line)
        (examples)
        (inputs)
        (outputs))
    (setf line (string-to-list (read-line in)))
    (setf num-examples (first line)
          num-inputs (second line)
          num-outputs (third line))

    (dotimes (n num-examples)
      (setf line (string-to-list (read-line in)))
      (setf inputs (subseq line 0 num-inputs)
            outputs (subseq line num-inputs))
      (push (list inputs outputs) examples))
    (reverse examples)))

(defun write-results (metrics filename)
  (with-open-file (out filename :direction :output
                                :if-exists :supersede
                                :if-does-not-exist :create)
    (setf metrics
          (mapcar #'(lambda (metric)
                      (let* ((a (first metric))
                             (b (second metric))
                             (c (third metric))
                             (d (fourth metric))
                             (accuracy (/ (+ a d) (+ a b c d)))
                             (precision (/ a (+ a b)))
                             (recall (/ a (+ a c)))
                             (f1 (/ (* 2 precision recall) (+ precision recall))))
                        (list a b c d accuracy precision recall f1)))
                  metrics))

    (format out "~{~{~d ~d ~d ~d ~,3f ~,3f ~,3f ~,3f~%~}~}"
            metrics)

    ;; micro-averaging
    (let* ((a (reduce #'+ metrics :key #'first))
           (b (reduce #'+ metrics :key #'second))
           (c (reduce #'+ metrics :key #'third))
           (d (reduce #'+ metrics :key #'fourth))
           (accuracy (/ (+ a d) (+ a b c d)))
           (precision (/ a (+ a b)))
           (recall (/ a (+ a c)))
           (f1 (/ (* 2 precision recall) (+ precision recall))))
      (format out "~,3f ~,3f ~,3f ~,3f~%" accuracy precision recall f1))

    ;; macro-averaging
    (let* ((num-classes (length metrics))
           (accuracy (/ (reduce #'+ metrics :key #'fifth) num-classes))
           (precision (/ (reduce #'+ metrics :key #'sixth) num-classes))
           (recall (/ (reduce #'+ metrics :key #'seventh) num-classes))
           (f1 (/ (* 2 precision recall) (+ precision recall))))
      (format out "~,3f ~,3f ~,3f ~,3f~%" accuracy precision recall f1))))

(defun set-inputs (network datum)
  "Set outputs of input layer neurons in NETWORK to DATUM. DATUM is in
  the form ((inputs) (outputs)) and has length 1. NETWORK is modified."
  (mapc #'(lambda (neuron x)
            (setf (neuron-a neuron) (coerce x 'double-float)))
        (car network) (car datum)))

(defun compute-layer-output (layer prev-layer)
  "Compute output of LAYER. LAYER is modified."
  (dolist (neuron layer)
    (setf (neuron-in neuron)
          (* (neuron-fixed-input neuron)
             (neuron-bias-weight neuron)))
    (map nil #'(lambda (input-neuron w)
                 (incf (neuron-in neuron)
                       (* w (neuron-a input-neuron))))
         prev-layer (neuron-in-weights neuron))
    (setf (neuron-a neuron)
          (funcall (neuron-g neuron) (neuron-in neuron)))))

(defun forpropagation (network)
  "Propogate inputs of NETWORK forward to compute outputs. NETWORK is
  modified."
  (labels ((forprop (net prev-net)
             (if net
                 (progn
                   (compute-layer-output (car net) (car prev-net))
                   (forprop (cdr net) (cdr prev-net))))))
    (forprop (cdr network) network)))

(defun backpropagation (network datum alpha)
  "Propagate inputs of NETWORK forward to compute outputs, then
   propagate deltas backward and update weights. NETWORK is modified."
  (labels
      ((backprop (net prev-net datum alpha &key (depth 0))
         (if (null net)
             ;; When net is null, we've reached the bottom and prev-net
             ;; contains only the output layer. Set output deltas.
             (mapc #'(lambda (neuron y)
                       (setf (neuron-delta neuron)
                             (* (funcall (neuron-dg neuron) (neuron-in neuron))
                                (- y (neuron-a neuron)))))
                   (car prev-net) (cadr datum))
             (progn
               ;; propagate inputs forward
               (compute-layer-output (car net) (car prev-net))
               (backprop (cdr net) (cdr prev-net) datum alpha :depth (1+ depth))

               ;; propagate deltas backward and update weights on the way back up
               (let ((layer (car prev-net))
                     (next-layer (car net)))
                 ;; compute deltas for layer if layer is not input layer
                 (if (/= depth 0)
                     (mapc #'(lambda (neuron i)
                               (setf (neuron-delta neuron)
                                     (* (funcall (neuron-dg neuron) (neuron-in neuron))
                                        (reduce #'+ next-layer
                                                :key #'(lambda (output-neuron)
                                                         (* (elt (neuron-in-weights output-neuron) i)
                                                            (neuron-delta output-neuron)))))))
                           layer (iota (length layer))))
                 ;; update input weights of next-layer
                 (dolist (output-neuron next-layer)
                   (incf (neuron-bias-weight output-neuron)
                         (* alpha
                            (neuron-fixed-input output-neuron)
                            (neuron-delta output-neuron)))
                   (setf (neuron-in-weights output-neuron)
                         (map '(vector double-float)
                              #'(lambda (w neuron)
                                  (+ w (* alpha
                                          (neuron-a neuron)
                                          (neuron-delta output-neuron))))
                              (neuron-in-weights output-neuron) layer))))))))
    (backprop (cdr network) network datum alpha :depth 0)))

(defun learn (network data &key (alpha 0.1d0) (epochs 100))
  "Train NETWORK on DATA using backpropogation. DATA is in the
   form ((inputs) (outputs)). NETWORK is modified."
  (do ((epoch 0 (1+ epoch)))
      ((= epoch epochs) network)
    (dolist (datum data)
      (set-inputs network datum)
      (backpropagation network datum alpha))))

(defun think (network data &key (boolean-output nil))
  "Apply (presumable trained) NETWORK to DATA. Data is in the
  form ((inputs) (outputs)). Return a list of lists, where each inner
  list contains the outputs for a datum. If the output data is
  epxected to be boolean, also return a list of lists, where each
  inner list is (A B C D), the confusion matrix values, for one of the
  output classes."
  (let ((results)
        (metrics))
    (dolist (datum data)
      ;; set input layer outputs
      (set-inputs network datum)
      ;; propogate inputs forward
      (forpropagation network)
      (push (mapcar #'neuron-a (car (last network)))
            results))
    (setf results (reverse results))

    (if boolean-output
        (let ((expected-results (mapcar #'cadr data)))
          (setf results (mapcar #'(lambda (res)
                                    (mapcar #'round res)) results))
          (setf metrics
                ; count each symbol (A, B, C, or D) for each class
                (mapcar #'(lambda (class-results)
                            (list (count 'a class-results)
                                  (count 'b class-results)
                                  (count 'c class-results)
                                  (count 'd class-results)))
                        (apply #'mapcar #'list    ; zip (transpose) lists
                               ;; replace outupt with symbol representing which count it
                               ;; should be added to (A, B, C, or D) for its class
                               (mapcar #'(lambda (result expected-result)
                                           (mapcar #'(lambda (r e-r)
                                                       (cond ((= 1 r e-r) 'a)
                                                             ((> r e-r) 'b)
                                                             ((< r e-r) 'c)
                                                             ((= 0 r e-r) 'd)))
                                                   result expected-result))
                                       results expected-results))))))
  (values results metrics)))
