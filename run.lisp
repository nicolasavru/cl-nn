(ql:quickload "cl-nn")
(in-package cl-nn)

(setf *random-state* (make-random-state t))

(let ((net-fname)
      (data-fname)
      (output-fname)
      (net)
      (data)
      (alpha)
      (epochs)
      (training-p))
  (setf training-p (y-or-n-p "Train a neural network?"))

  (format *query-io* "Enter the neural network file: ")
  (force-output *query-io*)
  (setf net-fname (read-line *query-io*))

  (format *query-io* "Enter the data file: ")
  (force-output *query-io*)
  (setf data-fname (read-line *query-io*))

  (format *query-io* "Enter the output file: ")
  (force-output *query-io*)
  (setf output-fname (read-line *query-io*))

  (setf net (load-network net-fname))
  (setf data (load-data data-fname))

  (if training-p
      (progn 
        (format *query-io* "Enter the learning rate to use: ")
        (force-output *query-io*)
        (setf alpha (read-from-string (read-line *query-io*)))

        (format *query-io* "Enter the number of epochs to simulate: ")
        (force-output *query-io*)
        (setf epochs (read-from-string (read-line *query-io*)))

        (learn net data :alpha alpha :epochs epochs)
        (write-network net output-fname))
      (multiple-value-bind (results metrics)
          (think net data :boolean-output t)
        (declare (ignore results))
        (write-results metrics output-fname))))


(sb-ext:exit)
