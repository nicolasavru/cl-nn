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
      (setf (aref network pos) (reverse layer))
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

(defun write-network (network filename)
  (with-open-file (out filename :direction :output
                                :if-exists :supersede
                                :if-does-not-exist :create)
    (let ((sizes (map 'list #'length network))
          (weights (map 'list
                        #'(lambda (layer)
                            (map 'list
                                 #'(lambda (neuron)
                                     (cons (neuron-bias-weight neuron)
                                           (coerce (neuron-in-weights neuron) 'list)))
                                 layer))
                        (subseq network 1))))       ; skip the input layer
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
    (reverse examples)))

(defun set-inputs (network datum)
  (map nil #'(lambda (neuron x)
               (setf (neuron-a neuron) x))
       (elt network 0) (car datum)))

(defun forward-prop (network)
  (let ((num-layers (length network)))
    (map-iota #'(lambda (n)
                  (let ((layer (elt network n))
                        (prev-layer (elt network (1- n)))
                        (i 0))
                    (dolist (neuron layer)
                      (setf (neuron-in neuron)
                            (* (neuron-fixed-input neuron)
                               (neuron-bias-weight neuron)))
                      (setf i 0)
                      (dolist (input-neuron prev-layer)
                        (incf (neuron-in neuron)
                              (* (elt (neuron-in-weights neuron) i)
                                 (neuron-a input-neuron)))
                        (incf i))
                      (setf (neuron-a neuron)
                            (funcall (neuron-g neuron) (neuron-in neuron))))))
              (1- num-layers) :start 1)))

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
      (forward-prop network)
      (push (map 'list #'neuron-a
                 (elt network (1- (length network)))) results))
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
                        (apply #'mapcar #'list              ; zip (transpose) lists
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

(defun learn (network training-data &key (alpha 0.1) (epochs 100))
  "Train NETWORK on TRAINING-DATA using back-propogation. DATA is in
   the form ((inputs) (outputs))."
  (do ((num-layers (length network))
       (epoch 0 (1+ epoch)))
      ((= epoch epochs) network)
    (dolist (datum training-data)
      ;; set input layer outputs
      (set-inputs network datum)
      ;; propogate inputs forward
      (forward-prop network)
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
                        (setf (neuron-delta neuron) 0.0)
                        (dolist (output-neuron next-layer)
                          (incf (neuron-delta neuron)
                                (* (elt (neuron-in-weights output-neuron) i)
                                   (neuron-delta output-neuron))))
                        (setf (neuron-delta neuron)
                              (* (funcall (neuron-dg neuron) (neuron-in neuron))
                                 (neuron-delta neuron)))
                      (incf i))
                      ;; update input weights of next-layer
                      (dolist (output-neuron next-layer)
                        (incf (neuron-bias-weight output-neuron)
                              (* alpha
                                 (neuron-fixed-input output-neuron)
                                 (neuron-delta output-neuron)))
                        (setf (neuron-in-weights output-neuron)
                              (map '(vector float)
                                   #'(lambda (w neuron)
                                       (+ w (* alpha
                                               (neuron-a neuron)
                                               (neuron-delta output-neuron))))
                                   (neuron-in-weights output-neuron) layer)))))
                (- num-layers 1) :start (- num-layers 2) :step -1))))
