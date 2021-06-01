from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk
import pickle
from itertools import cycle


class Window(object):
    def __init__(self):

        """Creation of main window using tkinter. This window will show all the results and the options to selection
        for training or prediction."""

        self.window = Tk(className='Discrimination of reflected sound signals')
        self.window.configure(background="#edfcf8")

        """ Label declaration for title and train and prediction sections"""

        title_label = Label(self.window, text="Discrimination of reflected sound signals")
        title_label.grid(row=0, column=1, columnspan=6, sticky='nsew')
        title_label.configure(background="#edfcf8", font='Helvetica 20 bold', height=3)

        sel_train_data = Label(self.window, text="Select Traning Data")
        sel_train_data.grid(row=1, column=0)
        sel_train_data.configure(background="#edfcf8")

        sel_predict_data = Label(self.window, text="Select Prediction Data")
        sel_predict_data.grid(row=3, column=0)
        sel_predict_data.configure(background="#edfcf8")

        predict_output = Label(self.window, text="Prediction Result", font='Helvetica 11 bold')
        predict_output.grid(row=6, columnspan=2, sticky='e')
        predict_output.configure(background="#edfcf8", height=2)

        train_validation = Label(self.window, text="Validation data output", font='Helvetica 11 bold')
        train_validation.grid(row=7, columnspan=3, sticky='e')
        train_validation.configure(background="#edfcf8", height=2)

        """" Prediction object values label declaration """

        predict_object_1 = Label(self.window, text="Object #1")
        predict_object_1.grid(row=5, column=2)
        predict_object_1.configure(background="#edfcf8")

        predict_object_2 = Label(self.window, text="Object #2")
        predict_object_2.grid(row=5, column=4)
        predict_object_2.configure(background="#edfcf8")

        predict_object_3 = Label(self.window, text="Object #3")
        predict_object_3.grid(row=5, column=6)
        predict_object_3.configure(background="#edfcf8")

        """" Prediction object values entry box declaration """

        self.predict_object_1_entry = Entry(self.window)
        self.predict_object_1_entry.grid(row=6, column=2)

        self.predict_object_2_entry = Entry(self.window)
        self.predict_object_2_entry.grid(row=6, column=4)

        self.predict_object_3_entry = Entry(self.window)
        self.predict_object_3_entry.grid(row=6, column=6)

        """Training output labels"""

        train_object_1 = Label(self.window, text="Object #1")
        train_object_1.grid(row=10, column=1)
        train_object_1.configure(background="#edfcf8")

        train_object_2 = Label(self.window, text="Object #2")
        train_object_2.grid(row=10, column=2)
        train_object_2.configure(background="#edfcf8")

        train_object_3 = Label(self.window, text="Object #3")
        train_object_3.grid(row=10, column=3)
        train_object_3.configure(background="#edfcf8")

        pred_count = Label(self.window, text="Test Object Counts", height=2)
        pred_count.grid(row=10, column=0, sticky='e')
        pred_count.configure(background="#edfcf8")

        accuracy = Label(self.window, text="Accuracy")
        accuracy.grid(row=11, column=0, sticky='e')
        accuracy.configure(background="#edfcf8")

        precision = Label(self.window, text="Precision")
        precision.grid(row=12, column=0, sticky='e')
        precision.configure(background="#edfcf8")

        tp = Label(self.window, text="True Positive (TP)")
        tp.grid(row=13, column=0, sticky='e')
        tp.configure(background="#edfcf8")

        tn = Label(self.window, text="True Negative (TN)")
        tn.grid(row=14, column=0, sticky='e')
        tn.configure(background="#edfcf8")

        fp = Label(self.window, text="False Positive (FP)")
        fp.grid(row=15, column=0, sticky='e')
        fp.configure(background="#edfcf8")

        fn = Label(self.window, text="False Negative (FN)")
        fn.grid(row=16, column=0, sticky='e')
        fn.configure(background="#edfcf8")

        tpr = Label(self.window, text="Sensitivity / True Positive Rate (TPR)")
        tpr.grid(row=17, column=0, sticky='e')
        tpr.configure(background="#edfcf8")

        tnr = Label(self.window, text="Specificity / True Negative Rate (TNR)")
        tnr.grid(row=18, column=0, sticky='e')
        tnr.configure(background="#edfcf8")

        f1 = Label(self.window, text="F1 Score")
        f1.grid(row=19, column=0, sticky='e')
        f1.configure(background="#edfcf8")

        fdr = Label(self.window, text="False Discovery Rate (FDR)")
        fdr.grid(row=20, column=0, sticky='e')
        fdr.configure(background="#edfcf8")

        npv = Label(self.window, text="Negative Predictive Value (NPV)")
        npv.grid(row=21, column=0, sticky='e')
        npv.configure(background="#edfcf8")

        """object 1 entry box declaration for output values for test samples"""

        # Test output prediction for Object 1
        self.pred_entry1 = Entry(self.window, width=10)
        self.pred_entry1.grid(row=10, column=1)

        # Accuracy value for Object 1
        self.acc_entry_obj1 = Entry(self.window, width=10)
        self.acc_entry_obj1.grid(row=11, column=1)

        # Precision value for Object 1
        self.prec_entry_obj1 = Entry(self.window, width=10)
        self.prec_entry_obj1.grid(row=12, column=1)

        # True Positivity for Object 1
        self.tp_entry_obj1 = Entry(self.window, width=10)
        self.tp_entry_obj1.grid(row=13, column=1)

        # True Negativity for Object 1
        self.tn_entry_obj1 = Entry(self.window, width=10)
        self.tn_entry_obj1.grid(row=14, column=1)

        # False positive for Object 1
        self.fp_entry_obj1 = Entry(self.window, width=10)
        self.fp_entry_obj1.grid(row=15, column=1)

        # False negative for Object 1
        self.fn_entry_obj1 = Entry(self.window, width=10)
        self.fn_entry_obj1.grid(row=16, column=1)

        # True Positive Rate for Object 1
        self.tpr_entry_obj1 = Entry(self.window, width=10)
        self.tpr_entry_obj1.grid(row=17, column=1)

        # True Negative Rate for Object 1
        self.tnr_entry_obj1 = Entry(self.window, width=10)
        self.tnr_entry_obj1.grid(row=18, column=1)

        # F1 Score for Object 1
        self.f1_entry_obj1 = Entry(self.window, width=10)
        self.f1_entry_obj1.grid(row=19, column=1)

        # False Discovery Rate for Object 1
        self.fdr_entry_obj1 = Entry(self.window, width=10)
        self.fdr_entry_obj1.grid(row=20, column=1)

        # Negative Predictive Value for Object 1
        self.npv_entry_obj1 = Entry(self.window, width=10)
        self.npv_entry_obj1.grid(row=21, column=1)

        """object 2 entry box declaration for output values for test samples"""

        # Test output prediction for Object 2
        self.pred_entry2 = Entry(self.window, width=10)
        self.pred_entry2.grid(row=10, column=2)

        # Accuracy value for object 2
        self.acc_entry_obj2 = Entry(self.window, width=10)
        self.acc_entry_obj2.grid(row=11, column=2)

        # Precision value for Object 2
        self.prec_entry_obj2 = Entry(self.window, width=10)
        self.prec_entry_obj2.grid(row=12, column=2)

        # True Positivity for Object 2
        self.tp_entry_obj2 = Entry(self.window, width=10)
        self.tp_entry_obj2.grid(row=13, column=2)

        # True Negative for Object 2
        self.tn_entry_obj2 = Entry(self.window, width=10)
        self.tn_entry_obj2.grid(row=14, column=2)

        # False Positive for Object 2
        self.fp_entry_obj2 = Entry(self.window, width=10)
        self.fp_entry_obj2.grid(row=15, column=2)

        # False Negative for Object 2
        self.fn_entry_obj2 = Entry(self.window, width=10)
        self.fn_entry_obj2.grid(row=16, column=2)

        # True Positive Rate for Object 2
        self.tpr_entry_obj2 = Entry(self.window, width=10)
        self.tpr_entry_obj2.grid(row=17, column=2)

        # True Negative Rate for Object 2
        self.tnr_entry_obj2 = Entry(self.window, width=10)
        self.tnr_entry_obj2.grid(row=18, column=2)

        # F1 Score for Object 2
        self.f1_entry_obj2 = Entry(self.window, width=10)
        self.f1_entry_obj2.grid(row=19, column=2)

        # False Discovery Rate for Object 2
        self.fdr_entry_obj2 = Entry(self.window, width=10)
        self.fdr_entry_obj2.grid(row=20, column=2)

        # Negative Predictive Value for Object 2
        self.npv_entry_obj2 = Entry(self.window, width=10)
        self.npv_entry_obj2.grid(row=21, column=2)

        """object 3 label and entry box declaration for output values for test samples"""

        # Test output prediction for Object 3
        self.pred_entry3 = Entry(self.window, width=10)
        self.pred_entry3.grid(row=10, column=3)

        # Accuracy value for object 3
        self.acc_entry_obj3 = Entry(self.window, width=10)
        self.acc_entry_obj3.grid(row=11, column=3)

        # Precision value for Object 3
        self.prec_entry_obj3 = Entry(self.window, width=10)
        self.prec_entry_obj3.grid(row=12, column=3)

        # True Positivity for Object 3
        self.tp_entry_obj3 = Entry(self.window, width=10)
        self.tp_entry_obj3.grid(row=13, column=3)

        # True Negative for Object 3
        self.tn_entry_obj3 = Entry(self.window, width=10)
        self.tn_entry_obj3.grid(row=14, column=3)

        # False Positive for Object 3
        self.fp_entry_obj3 = Entry(self.window, width=10)
        self.fp_entry_obj3.grid(row=15, column=3)

        # False Negative for Object 3
        self.fn_entry_obj3 = Entry(self.window, width=10)
        self.fn_entry_obj3.grid(row=16, column=3)

        # True Positive Rate for Object 3
        self.tpr_entry_obj3 = Entry(self.window, width=10)
        self.tpr_entry_obj3.grid(row=17, column=3)

        # True Negative Rate for Object 3
        self.tnr_entry_obj3 = Entry(self.window, width=10)
        self.tnr_entry_obj3.grid(row=18, column=3)

        # F1 Score for Object 3
        self.f1_entry_obj3 = Entry(self.window, width=10)
        self.f1_entry_obj3.grid(row=19, column=3)

        # False Discovery Rate for Object 3
        self.fdr_entry_obj3 = Entry(self.window, width=10)
        self.fdr_entry_obj3.grid(row=20, column=3)

        # Negative Predictive Value for Object 3
        self.npv_entry_obj3 = Entry(self.window, width=10)
        self.npv_entry_obj3.grid(row=21, column=3)

        """ Total accuracy after training"""

        total_acc = Label(self.window, text="Total accuracy", font='Helvetica 11 bold')
        total_acc.grid(row=22, column=0)
        total_acc.configure(background="#edfcf8")

        self.total_acc = Entry(self.window, width=40)
        self.total_acc.grid(row=22, column=1, columnspan=3, pady=5)

        """ Button declarations"""

        # Training data import button. Will call "import_win" function
        train_import = Button(self.window, text="Import", width=12, command=lambda: self.import_win(tag=1))
        train_import.grid(row=1, column=1, pady=5)
        train_import.configure(background="#c6f7ea")

        # Prediction data import button. Will call "import_win" function
        predict_import = Button(self.window, text="Import", width=12, command=lambda: self.import_win(tag=2))
        predict_import.grid(row=3, column=1, pady=5)
        predict_import.configure(background="#c6f7ea")

        # Button to refresh all the entries. Will call function "refresh"
        refresh = Button(self.window, text="Refresh", command=lambda: self.refresh())
        refresh.grid(row=23, column=5)
        refresh.configure(background="#c6f7ea")

        canvas = Canvas(self.window, width=200, height=100)
        canvas.grid(row=23, column=7)
        path = r"C:\Users\Raghav\Desktop\wav\\logo.jpg"
        img = ImageTk.PhotoImage(Image.open(path))
        canvas.create_image(0, 0, anchor=NW, image=img)

        print("*** Main window created ***")

        self.window.mainloop()

    def row_col_selection(self, tag, toplavel, filepath, model_filepath=None):

        """ This function will count the total number of rows and columns
        of the input file selected and pass the files and values further
        for training or prediction based on the tag values"""

        toplavel.destroy()      # Destroy the window from which it was called

        folder_path = os.path.dirname(filepath)

        input_file = pd.read_excel(filepath, header=None)

        selected_file = pd.DataFrame(input_file)

        print("*** Selected file shape {}  ***".format(selected_file.shape))

        count_rows = selected_file.shape[0]
        count_columns = selected_file.shape[1]

        if tag == 3:    # Tag value 3 is for training functionality
            self.initiate_train(filepath=filepath, folder_path=folder_path, count_rows=count_rows,
                                count_columns=count_columns, selected_file=selected_file)

        if tag == 4:    # Tag value 4 is for prediction functionality
            self.initiate_predict(filepath=filepath, folder_path=folder_path, model_filepath=model_filepath,
                                  count_rows=count_rows, count_columns=count_columns, selected_file=selected_file)

        print("*** Total rows : {} // total columns : {} in dataframe ***".format(count_rows, count_columns))

    def initiate_train(self, filepath, folder_path, count_rows, count_columns, selected_file):

        """This function display total number of columns and rows from
        file selected for training and provide training button"""

        self.display_text = Text(self.window, height=4, width=35, state='disabled')
        self.display_text.grid(row=10, column=4, rowspan=3, columnspan=3)
        display_text_input = "Total number of rows : {} \nTotal number of columns : {}\n".format(count_rows,
                                                                                                 count_columns)
        self.display_text.config(state='normal')
        self.display_text.insert(1.0, display_text_input)
        self.display_text.config(state='disabled')

        sel_col_train = Label(self.window, text="Select Columns")
        sel_col_train.grid(row=1, column=2)
        sel_col_train.configure(background="#edfcf8")

        sel_rows_train = Label(self.window, text="Select Rows")
        sel_rows_train.grid(row=2, column=2)
        sel_rows_train.configure(background="#edfcf8")

        col_start_train = Label(self.window, text="Start")
        col_start_train.grid(row=1, column=3)
        col_start_train.configure(background="#edfcf8")

        col_end_train = Label(self.window, text="End")
        col_end_train.grid(row=1, column=5)
        col_end_train.configure(background="#edfcf8")

        row_start_train = Label(self.window, text="Start")
        row_start_train.grid(row=2, column=3)
        row_start_train.configure(background="#edfcf8")

        row_end_train = Label(self.window, text="End")
        row_end_train.grid(row=2, column=5)
        row_end_train.configure(background="#edfcf8")

        self.col_start_train_entry = Entry(self.window)
        self.col_start_train_entry.grid(row=1, column=4, padx=10)

        self.col_end_train_entry = Entry(self.window)
        self.col_end_train_entry.grid(row=1, column=6, padx=10)

        self.row_start_train_entry = Entry(self.window)
        self.row_start_train_entry.grid(row=2, column=4, padx=10)

        self.row_end_train_entry = Entry(self.window)
        self.row_end_train_entry.grid(row=2, column=6, padx=10)

        self.col_start_train_entry.insert(0, 0)
        self.col_end_train_entry.insert(0, count_columns)
        self.row_start_train_entry.insert(0, 0)
        self.row_end_train_entry.insert(0, count_rows)

        print("*** File selection for training done ***")

        train = Button(self.window, text="Train", width=12,
                       command=lambda: self.predict_or_train_call(tag=3, count_rows=count_rows,
                                                                  count_columns=count_columns, filepath=filepath,
                                                                  folder_path=folder_path, selected_file=selected_file))
        train.grid(row=1, column=7)
        train.configure(background="#c6f7ea")

    def initiate_predict(self, filepath, folder_path, count_rows, count_columns, selected_file, model_filepath=None):

        """This function display total number of columns and rows from
        file selected for prediction and provide predict button"""

        self.display_text = Text(self.window, height=4, width=35, state='disabled')
        self.display_text.grid(row=10, column=4, rowspan=3, columnspan=3)
        display_text_input = "Total number of rows : {} \nTotal number of columns : {}\n".format(count_rows,
                                                                                                 count_columns)
        self.display_text.config(state='normal')
        self.display_text.insert(1.0, display_text_input)
        self.display_text.config(state='disabled')

        sel_col_predict = Label(self.window, text="Select Columns")
        sel_col_predict.grid(row=3, column=2)
        sel_col_predict.configure(background="#edfcf8")

        sel_rows_predict = Label(self.window, text="Select Rows")
        sel_rows_predict.grid(row=4, column=2)
        sel_rows_predict.configure(background="#edfcf8")

        col_start_predict = Label(self.window, text="Start")
        col_start_predict.grid(row=3, column=3)
        col_start_predict.configure(background="#edfcf8")

        col_end_predict = Label(self.window, text="End")
        col_end_predict.grid(row=3, column=5)
        col_end_predict.configure(background="#edfcf8")

        row_start_predict = Label(self.window, text="Start")
        row_start_predict.grid(row=4, column=3)
        row_start_predict.configure(background="#edfcf8")

        row_end_predict = Label(self.window, text="End")
        row_end_predict.grid(row=4, column=5)
        row_end_predict.configure(background="#edfcf8")

        self.col_start_predict_entry = Entry(self.window)
        self.col_start_predict_entry.grid(row=3, column=4, padx=10)

        self.col_end_predict_entry = Entry(self.window)
        self.col_end_predict_entry.grid(row=3, column=6, padx=10)

        self.row_start_predict_entry = Entry(self.window)
        self.row_start_predict_entry.grid(row=4, column=4, padx=10)

        self.row_end_predict_entry = Entry(self.window)
        self.row_end_predict_entry.grid(row=4, column=6, padx=10)

        self.col_start_predict_entry.insert(0, 0)
        self.col_end_predict_entry.insert(0, count_columns)
        self.row_start_predict_entry.insert(0, 0)
        self.row_end_predict_entry.insert(0, count_rows)

        print("*** File selection for prediction done ***")

        predict = Button(self.window, text="Predict", width=12,
                      command=lambda: self.predict_or_train_call(tag=4, count_columns=count_columns, count_rows=count_rows,
                                                                 filepath=filepath, folder_path=folder_path,
                                                                 model_file=model_filepath, selected_file=selected_file))
        predict.grid(row=3, column=7)
        predict.configure(background="#c6f7ea")

    def predict_or_train_call(self, tag, count_rows, count_columns, selected_file, filepath, folder_path, model_file=None):

        """ This function will slice the selected file as per the row and column selected"""

        print("*** File passed for sliceing: {} ***".format(filepath))

        xx_column = None
        yy_column = None
        xx_row = None
        yy_row = None

        if tag == 3:

            xx_column = self.col_start_train_entry.get()
            yy_column = self.col_end_train_entry.get()

            xx_row = self.row_start_train_entry.get()
            yy_row = self.row_end_train_entry.get()

            print("*** Slicing of file complete for training ***")

        if tag == 4:

            xx_column = self.col_start_predict_entry.get()
            yy_column = self.col_end_predict_entry.get()

            xx_row = self.row_start_predict_entry.get()
            yy_row = self.row_end_predict_entry.get()

            print("*** Slicing of file complete for prediction ***")

        selection_text_display = "Rows selected : {} to {}\nColumns selected : {} to {}".format(xx_row,
                                                                                                yy_row, xx_column,
                                                                                                yy_column)
        self.display_text.config(state='normal')
        self.display_text.insert(3.0, selection_text_display)
        self.display_text.config(state='disabled')

        dataframe = pd.DataFrame(selected_file.iloc[int(xx_row):int(yy_row), int(xx_column):int(yy_column)])

        print("*** Input Data transferred for MLP operation ***")

        self.mlp_initiation(dataframe=dataframe, model_file=model_file, tag=tag, folder_path=folder_path)

    def import_win(self, tag=None):

        """ This function will create new window where input file selection will be done """

        print("*** Input file selection initiated ***")

        self.top_train = Toplevel(self.window)
        self.top_train.title("Input file selection")
        self.top_train.configure(background="#edfcf8")

        self.train_filename = StringVar()
        self.model_filename = StringVar()
        self.model = None

        import_data_l1 = Label(self.top_train, text="Importing Data")
        import_data_l1.grid(row=0, column=2, padx=4, pady=2)
        import_data_l1.configure(background="#edfcf8")

        import_l2 = Label(self.top_train, text="Combining Multiple Data Files")
        import_l2.grid(row=1, column=0, padx=4, pady=2)
        import_l2.configure(background="#edfcf8")

        import_l3 = Label(self.top_train, text="Select Data File")
        import_l3.grid(row=2, column=0, padx=4, pady=2)
        import_l3.configure(background="#edfcf8")

        path_train_file = Entry(self.top_train, textvariable=self.train_filename)
        path_train_file.grid(row=2, column=2, padx=4, pady=2)

        browse_merge = Button(self.top_train, text="Merge files", width=12, command=lambda: MergeFiles(self.top_train))
        browse_merge.grid(row=1, column=3, padx=4, pady=2, columnspan=2)
        browse_merge.configure(background="#c6f7ea")

        browse_file_sel = Button(self.top_train, text="Browse", width=12, command=lambda: self.get_file_path(tag=tag))
        browse_file_sel.grid(row=2, column=3, padx=4, pady=2)
        browse_file_sel.configure(background="#c6f7ea")

    def get_file_path(self, tag):

        """ Selection of input file and if doing prediction then also trained model file"""

        picked_file = filedialog.askopenfilename(initialdir="/", title="Select input File",
                                                 filetypes=(("Excel Type1", "*.xlsx*"), ("other excel", "*.xls*")))

        self.train_filename.set(picked_file)
        self.picked_file_name = self.train_filename.get()

        if (tag == 1) and (not self.picked_file_name == False):     # Tag 1 for training

            print("*** Training input file selected ***")

            # Save button will show after selection of training input file
            save_train = Button(self.top_train, text="Save", width=12,
                                command=lambda: self.row_col_selection(filepath=self.picked_file_name,
                                                                       toplavel=self.top_train, tag=3))
            save_train.grid(row=3, column=0, padx=4, pady=2)
            save_train.configure(background="#c6f7ea")

        if tag == 2:    # Tag 2 for prediction
            import_l4 = Label(self.top_train, text="Select Trained Model File")
            import_l4.grid(row=3, column=0, padx=4, pady=2)
            import_l4.configure(background="#edfcf8")

            model_filename = Entry(self.top_train, textvariable=self.model_filename)
            model_filename.grid(row=3, column=2, padx=4, pady=2)

            browse_model_sel = Button(self.top_train, text="Browse", width=12,
                                      command=lambda: self.get_model_file(tag=tag))
            browse_model_sel.grid(row=3, column=3, padx=4, pady=2)
            browse_model_sel.configure(background="#c6f7ea")

    def get_model_file(self, tag=None):

        """Selection of trained model file"""

        model_file = filedialog.askopenfilename(initialdir="/", title="Select trained model file")

        self.model_filename.set(model_file)
        self.model_file_name = self.model_filename.get()

        if (tag == 2) and (not self.picked_file_name == False) and (not self.model_file_name == False):

            print(" *** Prediction input file and trained model file selected ***")

            # File save buttin will show after selection of both prediction input file and trained model file
            save_predict = Button(self.top_train, text="Save", width=12,
                               command=lambda: self.row_col_selection(filepath=self.picked_file_name,
                                                                      model_filepath=self.model_file_name,
                                                                      toplavel=self.top_train, tag=4))
            save_predict.grid(row=4, column=0, padx=4, pady=2)
            save_predict.configure(background="#c6f7ea")

    def mlp_initiation(self, dataframe=None, tag=None, folder_path=None, model_file=None):

        """ SKlearn MLP classifier will be called here and the parameters set"""

        print("*** MLP process started called ***")

        # Trained model filename and path defined.
        # Trained model file will saved in the same folder from where training file is selected
        pickle_filename = "{}\\finalized_model.pkl".format(folder_path)

        max_iter = 4000
        activation = "relu"
        mlp = MLPClassifier(max_iter=max_iter, activation=activation, learning_rate='adaptive')

        if tag == 3:    # Tag 3 for training
            self.train_mlp(dataframe=dataframe, folder_path=folder_path, pickle_filename=pickle_filename, mlp=mlp)
        if tag == 4:    # Tag 4 for prediction
            self.prediction_mlp(model_file=model_file, dataframe=dataframe, folder_path=folder_path)

    def train_mlp(self, dataframe, folder_path, pickle_filename, mlp):

        """ Traing process performed"""
        print("*** Training initiated ***")

        data_train = dataframe.drop(dataframe.columns[-1], axis=1)
        class_train = dataframe.iloc[:, -1:]

        # Splitting training input file in testing and training section
        x_train, x_test, y_train, y_test = train_test_split(data_train, class_train, test_size=0.2)

        # Performing training
        mlp.fit(x_train, y_train.values.ravel())
        pickle.dump(mlp, open(pickle_filename, 'wb'))

        print("*** pickle file saved in : {}    ***".format(pickle_filename))

        #Performing Prediction on testing data
        data_test = pd.DataFrame(x_test)
        class_test = pd.DataFrame(y_test).values.flatten()

        prediction = mlp.predict(data_test)

        values, counts = np.unique(prediction, return_counts=True)

        print("*** prediction completed and values={} and count{} ***".format(values, counts))

        #Creation of confusion matrix
        cm = sk.confusion_matrix(class_test, prediction)

        df = pd.DataFrame(data=cm, columns=values, index=values)

        # Measurement values calculation
        self.tp = {}
        self.fp = {}
        self.fn = {}
        self.tn = {}
        self.f1_local = {}
        self.accuracy_local = {}
        self.tpr = {}
        self.tnr = {}
        self.fdr = {}
        self.npv = {}
        self.ppv = {}
        self.fnr = {}
        self.fpr = {}
        self.fomr = {}

        for label in values:
            self.tp[label] = df.loc[label, label]
            self.fp[label] = df[label].sum() - self.tp[label]
            self.fn[label] = df.loc[label].sum() - self.tp[label]
            tp, fp, fn = self.tp[label], self.fp[label], self.fn[label]

            self.ppv[label] = tp / (tp + fp) if (tp + fp) > 0. else 0.
            self.tpr[label] = tp / (tp + fn) if (tp + fp) > 0. else 0.
            p, r = self.ppv[label], self.tpr[label]

            self.f1_local[label] = 2. * p * r / (p + r) if (p + r) > 0. else 0.
            self.accuracy_local[label] = tp / (tp + fp + fn) if (tp + fp + fn) > 0. else 0.
            self.fdr[label] = 1 - p
            self.fnr[label] = 1 - self.tpr[label]

        print("#-- Local measures --#")
        print("*** True Positives:", self.tp," ***")
        print("*** False Positives:", self.fp," ***")
        print("*** False Negatives:", self.fn," ***")
        print("*** Precision:", self.ppv," ***")
        print("*** Recall:", self.tpr," ***")
        print("*** F1-Score:", self.f1_local," ***")
        print("*** Accuracy:", self.accuracy_local," ***")
        print("*** fdr : ", self.fdr," ***")
        print("*** fnr : ", self.fnr," ***")

        for label in set(class_test):
            self.tn[label] = len(class_test) - (self.tp[label] + self.fp[label] + self.fn[label])

            self.tnr[label] = self.tn[label] / (self.tn[label] + self.fp[label])
            self.npv[label] = self.tn[label] / (self.tn[label] + self.fn[label])
            self.fpr[label] = 1 - self.tnr[label]
            self.fomr[label] = 1 - self.npv[label]

        print("*** True Negatives:", self.tn," ***")
        print("*** tnr : ", self.tnr," ***")
        print("*** fpr : ", self.fpr," ***")
        print("*** npv : ", self.npv," ***")
        print("*** fomr : ", self.fomr," ***")

        # Overall accuracy
        self.acc_total = sk.accuracy_score(class_test, prediction)

        print("*** Confusion matrix successfully ran and assess values generated ***")

        # output plot label
        output_plot = Label(self.window, text="Output Plots")
        output_plot.grid(row=13, column=5, sticky='e')
        output_plot.configure(background="#edfcf8")

        #Button for display of confusion matrix
        confu_matrix = Button(self.window, text="Confusion Matrix", width=12,
                              command=lambda: self.confusion_matrix(mlp=mlp, data_test=data_test,
                                                                    class_test=class_test))

        confu_matrix.grid(row=17, column=5)
        confu_matrix.configure(background="#c6f7ea")

        # Button for display of ROC plot
        roc_plot = Button(self.window, text="ROC Plot", width=12,
                          command=lambda: self.roc_plot(class_test=class_test, prediction=prediction))
        roc_plot.grid(row=19, column=5)
        roc_plot.configure(background="#c6f7ea")

        self.train_complete(values=values, counts=counts)

    def confusion_matrix(self, mlp, data_test, class_test):
        """ Display confusion matrix"""

        print("*** Generating Confusion matrix plot ***")
        disp = sk.plot_confusion_matrix(mlp, data_test, class_test, cmap=plt.cm.Blues)
        plt.show()

    def roc_plot(self, class_test, prediction):

        """ Display ROC plot"""

        print("*** Generating ROC plot ***")

        classes = 3

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(classes):
            roc_parm1 = np.array(pd.get_dummies(class_test))[:, i]
            roc_parm2 = np.array(pd.get_dummies(prediction))[:, i]

            fpr[i], tpr[i], _ = sk.roc_curve(roc_parm1,
                                             roc_parm2)
            roc_auc[i] = sk.auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sk.auc(fpr["macro"], tpr["macro"])

        lw = 2
        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
            print("roc_auc_score of object ", i + 1, ": ", roc_auc[i])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        print("*** ROC plot generation complete ***")

    def train_complete(self, values, counts):

        """ create window informing completion of training"""

        top_train = Toplevel(self.window)
        top_train.title("Training Complete")
        top_train.configure(background="#edfcf8")

        print("*** Training complete values will be passed to access value display ***")

        l1 = Label(top_train, text="Model Trained Successfully")
        l1.grid(row=0, column=0)
        l1.configure(background="#edfcf8")

        b1 = Button(top_train, text="OK", width=12, command=lambda: self.train_output(tn=self.tn, tp=self.tp,
                                                                                      fp=self.fp,
                                                                                      fn=self.fn, tpr=self.tpr,
                                                                                      tnr=self.tnr,
                                                                                      f1_score=self.f1_local,
                                                                                      fdr=self.fdr,
                                                                                      npv=self.npv,
                                                                                      acc_local=self.accuracy_local,
                                                                                      preci=self.ppv, values=values,
                                                                                      ok_win=top_train, counts=counts,
                                                                                      acc_total=self.acc_total))
        b1.grid(row=1, column=0)
        b1.configure(background="#c6f7ea")

    def train_output(self, acc_local, preci, tp, tn, fp, fn, tpr, tnr, f1_score, fdr, npv, ok_win, values, counts,
                     acc_total):

        """Display training measurements"""

        print("*** Values received for display ***")

        for label in values:
            if label == 'Object #1':
                self.acc_entry_obj1.insert(0, round(acc_local[label], 2))
                self.prec_entry_obj1.insert(0, round(preci[label], 2))
                self.tp_entry_obj1.insert(0, round(tp[label], 2))
                self.tn_entry_obj1.insert(0, round(tn[label], 2))
                self.fp_entry_obj1.insert(0, round(fp[label], 2))
                self.fn_entry_obj1.insert(0, round(fn[label], 2))
                self.tpr_entry_obj1.insert(0, round(tpr[label], 2))
                self.tnr_entry_obj1.insert(0, round(tnr[label], 2))
                self.f1_entry_obj1.insert(0, round(f1_score[label], 2))
                self.fdr_entry_obj1.insert(0, round(fdr[label], 2))
                self.npv_entry_obj1.insert(0, round(npv[label], 2))

            if label == 'Object #2':
                self.acc_entry_obj2.insert(0, round(acc_local[label], 2))
                self.prec_entry_obj2.insert(0, round(preci[label], 2))
                self.tp_entry_obj2.insert(0, round(tp[label], 2))
                self.tn_entry_obj2.insert(0, round(tn[label], 2))
                self.fp_entry_obj2.insert(0, round(fp[label], 2))
                self.fn_entry_obj2.insert(0, round(fn[label], 2))
                self.tpr_entry_obj2.insert(0, round(tpr[label], 2))
                self.tnr_entry_obj2.insert(0, round(tnr[label], 2))
                self.f1_entry_obj2.insert(0, round(f1_score[label], 2))
                self.fdr_entry_obj2.insert(0, round(fdr[label], 2))
                self.npv_entry_obj2.insert(0, round(npv[label], 2))

            if label == 'Object #3':
                self.acc_entry_obj3.insert(0, round(acc_local[label], 2))
                self.prec_entry_obj3.insert(0, round(preci[label], 2))
                self.tp_entry_obj3.insert(0, round(tp[label], 2))
                self.tn_entry_obj3.insert(0, round(tn[label], 2))
                self.fp_entry_obj3.insert(0, round(fp[label], 2))
                self.fn_entry_obj3.insert(0, round(fn[label], 2))
                self.tpr_entry_obj3.insert(0, round(tpr[label], 2))
                self.tnr_entry_obj3.insert(0, round(tnr[label], 2))
                self.f1_entry_obj3.insert(0, round(f1_score[label], 2))
                self.fdr_entry_obj3.insert(0, round(fdr[label], 2))
                self.npv_entry_obj3.insert(0, round(npv[label], 2))

        self.pred_entry1.insert(0, counts[0])
        self.pred_entry2.insert(0, counts[1])
        self.pred_entry3.insert(0, counts[2])

        self.total_acc.insert(0, round(acc_total, 2))
        ok_win.destroy()

        print("*** Training output displayed ***")

    def prediction_mlp(self, dataframe, model_file, folder_path):

        """ Prediction of number of objects in input file"""

        print("*** Trained model file = {}  ***".format(model_file))
        print("*** Prediction input data shape = {}  ***".format(dataframe.shape))

        with open(model_file, "rb") as sav_file:
            loaded_model = pickle.load(sav_file)

        prediction_result = loaded_model.predict(dataframe)

        values, counts = np.unique(prediction_result, return_counts=True)
        print("*** unique objects and there counts {} {} ***".format(values, counts))

        print("*** Prediction process is complete ***")

        # Button to display prediction results
        view_output = Button(self.window, text="View output", width=12,
                             command=lambda: self.view_output(prediction_result=prediction_result, dataframe=dataframe,
                                                              folder_path=folder_path))
        view_output.grid(row=15, column=5)
        view_output.configure(background="#c6f7ea")

        self.prediction_complete(values=values, counts=counts)

    def prediction_complete(self, values, counts):

        """ create window informing completion of prediction process"""

        print("*** Prediction complete ready for display ***")

        top_prediction = Toplevel(self.window)
        top_prediction.title("Prediction Complete")
        top_prediction.configure(background="#edfcf8")

        l1 = Label(top_prediction, text="Prediction Completed")
        l1.grid(row=0, column=0)
        l1.configure(background="#edfcf8")

        b1 = Button(top_prediction, text="OK", width=12, command=lambda: self.prediction_output(values=values, counts=counts,
                                                                                                prediction_win=top_prediction))
        b1.grid(row=1, column=0)
        b1.configure(background="#c6f7ea")

    def prediction_output(self, values, counts, prediction_win):

        """ Display prediction output"""

        print("*** Displaying number of object values ***")

        val = np.array(values)
        val_unique = np.unique(val)

        temp_dict = dict(zip(val_unique,counts))

        for (key, value) in temp_dict.items():
            if key == 'Object #1':
                self.predict_object_1_entry.insert(0, value)
            if key == 'Object #2':
                self.predict_object_2_entry.insert(0, value)
            if key == 'Object #3':
                self.predict_object_3_entry.insert(0, value)

        prediction_win.destroy()
        print("*** Prediction output values displayed ***")

    def view_output(self, prediction_result, dataframe, folder_path):

        """ View output of prediction for every row in a new window and save the result in a Excel file"""

        top_prediction_out = Toplevel(self.window)
        top_prediction_out.title("Prediction Complete")
        top_prediction_out.configure(background="#edfcf8")

        output_df = pd.DataFrame()
        scrollbar = Scrollbar(top_prediction_out)
        scrollbar.pack(side=RIGHT, fill=Y)

        mylist = Listbox(top_prediction_out, width=50, yscrollcommand=scrollbar.set)

        for i in range(dataframe.shape[0]):
            j = i+1
            obj_val = prediction_result[i]
            mylist.insert(END, "Row Number {} is object type {}".format(j, obj_val))

            output_df = output_df.append(dataframe.loc[i],ignore_index=True)
            row_index = output_df.index[i]
            output_df.loc[row_index, "obj_type"] = prediction_result[i]

        mylist.pack(side=LEFT, fill=BOTH)
        scrollbar.config(command=mylist.yview)

        # Output results will be saved on a excel file in the folder from where input file is selected
        output_df.to_excel(folder_path + '\\prediction_result.xlsx', index=False, header=None)

        print("*** Result saved in {}   ***".format(folder_path))

    def refresh(self):
        self.window.destroy()
        self.__init__()


class MergeFiles(object):
    """ This class is used to merge multiple excel files and add new column at the end of file with object values.
        This will also delete the first 6 columns of the excel sheet as the excel files provided don't need these
        first 6 columns. """

    def __init__(self, master):

        """ Creating window for the creation of window for merging of files."""

        self.top_merge = Toplevel(master)
        self.top_merge.title("mention a title")
        self.top_merge.configure(background="#edfcf8")

        self.folderPath = StringVar()
        self.object_name = []
        self.file_name = []
        self.f_name = {}
        self.f_object = {}

        select_folder = Label(self.top_merge, text="Select folder")
        select_folder.grid(row=0, column=0, padx=10, pady=20)
        select_folder.configure(background="#edfcf8")

        path_folder = Entry(self.top_merge, textvariable=self.folderPath)
        path_folder.grid(row=0, column=1, columnspan=3, padx=10, pady=20)

        btn_find = Button(self.top_merge, text="Browse", command=self.get_folder_path)
        btn_find.grid(row=0, column=5, padx=10, pady=20)
        btn_find.configure(background="#c6f7ea")

        # Button to display files in a folder. It calls the "list_files" function
        search_files = Button(self.top_merge, text="find all files", command=self.list_files)
        search_files.grid(row=0, column=6, padx=10, pady=20)
        search_files.configure(background="#c6f7ea")

    def get_folder_path(self):
        folder_selected = filedialog.askdirectory()
        self.folderPath.set(folder_selected)

    def list_files(self):

        """ list all files on the selected folder and provides checkboxes
            to select files and entry boxes for object value entries. """

        i = 1
        var_name = []

        merge = Button(self.top_merge, text="merge files", command=lambda: self.merge_files())
        merge.grid(row=0, column=12, padx=10, pady=20)
        merge.configure(background="#c6f7ea")

        for r, d, f in os.walk(self.folderPath.get()):
            for file in f:
                z = i - 1
                self.file_name.append(file)
                var_name.append("var_{}".format(i))
                self.object_name.append("name_{}".format(i))

                object_number_label = Label(self.top_merge, text="      enter the object number")
                object_number_label.grid(row=i, column=1)
                object_number_label.configure(background="#edfcf8")

                # Enter object names( only integer values of 1 or 2 or 3)
                self.object_name[z] = Entry(self.top_merge, state="disabled")

                self.object_name[z].grid(row=i, column=2)

                var_name[z] = IntVar()
                Checkbutton(self.top_merge, text=self.file_name[z], variable=var_name[z],
                            command=lambda v=var_name[z], e=self.object_name[z]: self.selection(v, e)) \
                    .grid(row=i, sticky=W)
                self.f_name[self.file_name[z]] = var_name[z]

                i += 1

    def selection(self, var, entry):

        if var.get() == 1:
            entry.configure(state="normal")
        else:
            entry.configure(state="disabled")

    def merge_files(self):

        """ Merge files together into a single file"""

        key_f_object = []
        list_object = []
        checker = []
        skipp_list = []
        all_data = pd.DataFrame()

        for x in range(len(self.object_name)):
            self.f_object[self.file_name[x]] = self.object_name[x].get()

            if self.object_name[x].get():
                list_object.append(self.object_name[x].get())
                checker.append("value")
            else:
                checker.append("empty")
                list_object.append('')

        for xx in range(len(list_object)):

            checker_condition = checker[xx]
            class_label = list_object[xx]

            for (key, value) in self.f_object.items():
                if value == class_label:
                    if key not in key_f_object:
                        key_f_object.append(key)
                    else:
                        continue

            for (name, var) in self.f_name.items():

                if name in skipp_list:
                    continue
                if (var.get() == 1) and (name in key_f_object):

                    skipp_list.append(name)

                    loc = self.folderPath.get() + "\\" + name
                    wb = pd.DataFrame(pd.read_excel(loc, header=None))

                    if checker_condition == "empty":
                        pass
                    elif checker_condition == "value":
                        wb = wb.assign( obj_column=class_label)

                    all_data = all_data.append(wb, ignore_index=True)

        all_data.to_excel(self.folderPath.get() + '\\temp_merged.xlsx', index=False, header=None)

        print("*** Deleting first 6 columns which are not required ***")

        col_removed_file = openpyxl.load_workbook(self.folderPath.get() + '\\temp_merged.xlsx')
        select_sheet = col_removed_file.active
        select_sheet.delete_cols(idx=1, amount=6)
        col_removed_file.save(filename='%s\\consolidated_data.xlsx' % self.folderPath.get())
        os.remove('%s\\temp_merged.xlsx' % self.folderPath.get())

        self.merged_file_path = "{}//consolidated_data.xlsx".format(self.folderPath.get())

        print("*** Mergeing of files completed ***")
        print("*** Merged file path {}  ***".format(self.merged_file_path))


Window()
