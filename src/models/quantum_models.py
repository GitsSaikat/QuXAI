# src/models/quantum_models.py

import numpy as np
import pennylane as qml

def sample_quantum_feature_map(x, num_qubits):
    """
    Basic encoding where each feature => RX rotation on each qubit.
    """
    for i in range(num_qubits):
        qml.RX(x[i], wires=i)

def train_qrf(X_train, y_train, feature_map=sample_quantum_feature_map):
    """
    Quantum Random Forest:
      - Convert each row to a quantum state
      - Train a RandomForest on amplitude vectors
    """
    from sklearn.ensemble import RandomForestClassifier
    
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()

    # Convert training samples to amplitude vectors
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_q, y_train)

    # Attach relevant attributes
    rf.num_qubits = num_qubits
    rf.quantum_feature_map = feature_map
    return rf

def train_qsvm(X_train, y_train, feature_map=sample_quantum_feature_map):
    """
    Quantum SVM:
      - Build NxN fidelity matrix using the quantum kernel
      - Train SVC(kernel='precomputed')
    """
    from sklearn.svm import SVC
    
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def feature_map_circuit(x):
        feature_map(x, num_qubits)
        return qml.state()

    def kernel_fn(x1, x2):
        state1 = feature_map_circuit(x1)
        state2 = feature_map_circuit(x2)
        return np.abs(np.dot(state1.conj(), state2)) ** 2

    train_kernel = qml.kernels.square_kernel_matrix(X_train, kernel_fn)

    svm = SVC(kernel='precomputed')
    svm.fit(train_kernel, y_train)

    # Attach relevant attributes
    svm.num_qubits = num_qubits
    svm.quantum_feature_map = feature_map
    return svm

def train_qknn(X_train, y_train, feature_map=sample_quantum_feature_map, n_neighbors=3):
    """
    Quantum KNN:
      - Build NxN fidelity matrix
      - distance = 1 - fidelity
      - KNeighborsClassifier(metric='precomputed')
    """
    from sklearn.neighbors import KNeighborsClassifier

    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def feature_map_circuit(x):
        feature_map(x, num_qubits)
        return qml.state()

    def kernel_fn(x1, x2):
        state1 = feature_map_circuit(x1)
        state2 = feature_map_circuit(x2)
        return np.abs(np.dot(state1.conj(), state2)) ** 2

    # NxN fidelity matrix
    train_kernel = qml.kernels.square_kernel_matrix(X_train, kernel_fn)

    # Distances
    dist_matrix = 1 - train_kernel
    dist_matrix = np.abs(dist_matrix)  # clamp negativity if needed

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
    knn.fit(dist_matrix, y_train)

    # Attach attributes
    knn.num_qubits = num_qubits
    knn.quantum_feature_map = feature_map
    return knn

def train_qlogistic(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum Logistic Regression using amplitude features
    from sklearn.linear_model import LogisticRegression
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = LogisticRegression()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qdt(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum Decision Tree using amplitude features
    from sklearn.tree import DecisionTreeClassifier
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = DecisionTreeClassifier()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qnb(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum Naive Bayes using amplitude features
    from sklearn.naive_bayes import GaussianNB
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = GaussianNB()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qada(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum AdaBoost using amplitude features
    from sklearn.ensemble import AdaBoostClassifier
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = AdaBoostClassifier()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qgb(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum Gradient Boosting using amplitude features
    from sklearn.ensemble import GradientBoostingClassifier
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = GradientBoostingClassifier()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qlda(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum Linear Discriminant Analysis using amplitude features
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qperceptron(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum Perceptron using amplitude features
    from sklearn.linear_model import Perceptron
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = Perceptron()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qridge(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum Ridge Classifier using amplitude features
    from sklearn.linear_model import RidgeClassifier
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = RidgeClassifier()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qsvc_poly(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum SVM with a polynomial kernel built from quantum fidelity
    from sklearn.svm import SVC
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def feature_map_circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    
    def kernel_fn(x1, x2):
        base = np.abs(np.dot(feature_map_circuit(x1).conj(), feature_map_circuit(x2))) ** 2
        return (base + 1) ** 3  # polynomial kernel of degree 3

    train_kernel = qml.kernels.square_kernel_matrix(X_train, kernel_fn)
    clf = SVC(kernel='precomputed')
    clf.fit(train_kernel, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf

def train_qextra(X_train, y_train, feature_map=sample_quantum_feature_map):
    # Quantum Extra Trees using amplitude features
    from sklearn.ensemble import ExtraTreesClassifier
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x)) for x in X_train])
    clf = ExtraTreesClassifier()
    clf.fit(X_train_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    return clf
