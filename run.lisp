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
      (training-p)
      (sizes))
  (setf training-p (y-or-n-p "Train a neural network?"))

  (if (and training-p (y-or-n-p "Generate a neural network?"))
      (progn
          (format *query-io* "Enter the neural network output file: ")
          (force-output *query-io*)
          (setf net-fname (read-line *query-io*))

          (format *query-io* "Enter a list of sizes for the network (ex: (3 2 3)): ")
          (force-output *query-io*)
          (setf sizes (read-from-string (read-line *query-io*)))

          (setf net (make-network sizes))
          (write-network net net-fname))
      (progn
        (format *query-io* "Enter the neural network input file: ")
        (force-output *query-io*)
        (setf net-fname (read-line *query-io*))
        (setf net (load-network net-fname))))

  (format *query-io* "Enter the data file: ")
  (force-output *query-io*)
  (setf data-fname (read-line *query-io*))
  (setf data (load-data data-fname))

  (format *query-io* "Enter the output file: ")
  (force-output *query-io*)
  (setf output-fname (read-line *query-io*))

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
