
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# Create a class to extract a column of data
# Put all the functions we created into a class


class DataExtractor(object):
    def __init__(self, data_category, column_name):

        header_file = 'header-'+data_category+'.csv'
        data_file = data_category+'.csv'

        '''
        if data_category=='A':
            header_file='header-A.csv'
            data_file='A.csv'
        elif data_category =='R':
            header_file='header-R.csv'
            data_file='R.csv'
        '''

        headers = open(header_file).read().split(',')
        index = headers.index(column_name)

        self.data = []
        for line in open(data_file):
            entry = line.split(',')[index]
            cleaned_entry = self.clean(entry)
            self.data.append(cleaned_entry)

    def clean(self, entry):
        # Not doing anything for generic column
        return entry

    def stats(self):
        counter = {}
        for entry in self.data:
            counter.setdefault(entry, 0)
            counter[entry] = counter[entry]+1

        for entry in counter:
            counter[entry] = '%.2f' % (counter[entry]*100.0/len(self.data))

        return counter

# TODO
# create a class inherits from DataExtractor
# overwrite the clean function to do the actual cleaning


class PurposeExtractor(DataExtractor):

    def clean(self, entry):
        # slowly add
        corrections = {'': 'other',
                       'debt consolidation': 'debt_consolidation',
                       'major purchase': 'major_purchase',
                       'credit card refinancing': 'credit_card',
                       'moving and relocation': 'moving',
                       'business loan': 'business',
                       'small_business': 'business',
                       'home improvement': 'home_improvement',
                       'renewable_energy': 'other',
                       'green loan': 'other'
                       }
        # conver to lower case
        key = entry.lower()

        # if it needs to be corrected, return the corrected value
        if key in corrections.keys():
            return corrections[key]
        else:
            return key


class NumberExtractor(DataExtractor):
    def clean(self, entry):
        try:
            return float(entry)
        except ValueError:
            return None


class EmploymentExtractor(DataExtractor):
    def clean(self, entry):

        # let's convert the value from string(text) into a number
        mapping = {'1 year': 1,
                   '10+ years': 10,
                   '2 years': 2,
                   '3 years': 3,
                   '4 years': 4,
                   '5 years': 5,
                   '6 years': 6,
                   '7 years': 7,
                   '8 years': 8,
                   '9 years': 9,
                   '< 1 year': 0}

        # CHALLENGE
        # typing this mapping line by line is quite tedious and error prone
        # Write a loop to initialize some values, automate those can be automated
        if entry in mapping:
            return mapping[entry]
        else:
            # Missing or wrong data is not used, we set them to be empty
            return None
        return entry


def exclude_empty(X, Y):
    filtered_x = []
    filtered_y = []

    for index in range(0, len(X)):
        x = X[index]
        y = Y[index]

        # both x, y need to be valid
        # ignore those empty entries

        if x is None or y is None:
            continue

        # We notice there are a few entries with very high loan values
        # The are called the outliers, let's manually remove them

        if x > 39000 or y > 39000:
            continue
        else:
            filtered_x.append(x)
            filtered_y.append(y)

    # CHALLENGE
    # use zip() function to simplify this loop
    # https://www.programiz.com/python-programming/methods/built-in/zip

    # convert it into numpy array to take advantage of numpy functions
    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)

    return filtered_x, filtered_y


emp_a = EmploymentExtractor('A', 'emp_length')
amnt_a = NumberExtractor('A', 'loan_amnt')

emp_r = EmploymentExtractor('R', 'emp_length')
amnt_r = NumberExtractor('R', 'loan_amnt')


# Preparing the data and groups for SVM
# we are building something like this
# data=  [apple, kale, banana, kale]
# groups=[fruit, veg,  fruit,  veg]

ae, aa = exclude_empty(emp_a.data, amnt_a.data)
re, ra = exclude_empty(emp_r.data, amnt_r.data)

groups = []
data = []
# alternate the data
for index in range(0, len(ae)):
    data.append([ae[index], aa[index]/4000])
    data.append([re[index], ra[index]/4000])
    groups.append('g')
    groups.append('r')
data = np.array(data)
colors = np.array(groups)
training_sample_size = int(len(data)*0.8)
print(training_sample_size, len(data))


# we take the first 90% entries as training data using list slicing method [start:end]
X_train = data[:training_sample_size]
y_train = colors[:training_sample_size]
X_test = data[training_sample_size:]
y_test = colors[training_sample_size:]

# fit the model

clf = svm.SVC(max_iter=1000)
clf.fit(X_train, y_train)

plt.figure(0)
plt.clf()
plt.scatter(data[:, 0], data[:, 1], c=colors, zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)

# Circle out the test data
plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10, edgecolor='k')

plt.axis('tight')
x_min = data[:, 0].min()
x_max = data[:, 0].max()
y_min = data[:, 1].min()
y_max = data[:, 1].max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
plt.show()
