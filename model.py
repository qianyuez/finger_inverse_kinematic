import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import keras.backend as K
import cv2


image_size = (300, 300)
goal_radius = 10
goal = (0, 0)
l0 = 120.0
l1 = 90.0
l2 = 60.0
x0 = 10
y0 = 150
actions = [1, 1]
point0 = {'x': 10.0, 'y': 150.0}
point1 = {'x': point0['x'] + l0, 'y': point0['y']}
point2 = {'x': point1['x'] + l1, 'y': point1['y']}
point3 = {'x': point2['x'] + l2, 'y': point2['y']}
learning_rate = 0.001

kinemitic_model = Sequential()
kinemitic_model.add(Dense(64, input_dim=2, activation='relu'))
kinemitic_model.add(Dense(64, activation='relu'))
kinemitic_model.add(Dense(2, activation='sigmoid'))
kinemitic_model.summary()

def position_loss(y_true, y_pred):
    goal = y_true[0]
    action = y_pred[0]
    actions = action
    theta0 = 0.5 * np.pi * (action[0] - 0.5)
    theta1 = -0.5 * np.pi * action[1]
    theta2 = theta1

    x1 = x0 + l0 * K.cos(theta0)
    y1 = y0 - l0 * K.sin(theta0)

    x2 = x1 + l1 * K.cos(theta1 + theta0)
    y2 = y1 - l1 * K.sin(theta1 + theta0)

    x3 = x2 + l2 * K.cos(theta2 + theta1 + theta0)
    y3 = y2 - l2 * K.sin(theta2 + theta1 + theta0)

    x3 /= image_size[0]
    y3 /= image_size[1]
    goal_x = goal[0] / image_size[0]
    goal_y = goal[1] / image_size[1]

    loss = K.square(x3 - goal_x) + K.square(y3 - goal_y)
    return loss

kinemitic_model.compile(loss=position_loss, optimizer=Adam(lr=learning_rate))


def update_position(action):
    theta0 = 0.5 * np.pi * (action[0] - 0.5)
    theta1 = -0.5 * np.pi * action[1]
    theta2 = theta1
    point1['x'] = point0['x'] + l0 * np.cos(theta0)
    point1['y'] = point0['y'] - l0 * np.sin(theta0)

    point2['x'] = point1['x'] + l1 * np.cos(theta1 + theta0)
    point2['y'] = point1['y'] - l1 * np.sin(theta1 + theta0)

    point3['x'] = point2['x'] + l2 * np.cos(theta2 + theta1 + theta0)
    point3['y'] = point2['y'] - l2 * np.sin(theta2 + theta1 + theta0)

def onMouse(event, x, y, flags, param):
    global goal
    goal = (x, y)

def render():
    img = np.zeros((image_size[0], image_size[1], 3), np.uint8)
    img.fill(255)

    cv2.line(img, (int(point0['x']), int(point0['y'])), (int(point1['x']), int(point1['y'])),
             (120, 30, 30), 5)
    cv2.line(img, (int(point1['x']), int(point1['y'])), (int(point2['x']), int(point2['y'])),
             (120, 30, 30), 5)
    cv2.line(img, (int(point2['x']), int(point2['y'])), (int(point3['x']), int(point3['y'])),
             (120, 30, 30), 5)
    cv2.circle(img, (int(goal[0]), int(goal[1])), int(goal_radius), (120, 120, 120), -1, 8)

    cv2.imshow('image', img)
    cv2.waitKey(1)

batch_size = 32
index = 0
kinemitic_model.load_weights('model')

cv2.namedWindow('image')
cv2.setMouseCallback('image', onMouse)

while True:
    #goal = (np.random.uniform(0, image_size[0]), np.random.uniform(0, image_size[1]))
    goal_input = np.array([goal[0] / image_size[0], goal[1] / image_size[1]], dtype=np.float).reshape(1, 2)
    action = kinemitic_model.predict(goal_input)
    action = action[0]
    update_position(action)
    #print(str((goal[0]/image_size[0] - point3['x']/image_size[0])**2 + (goal[1]/image_size[1] - point3['y']/image_size[1])**2))
    #loss = kinemitic_model.train_on_batch(goal_input, np.array([goal[0], goal[1]], dtype=np.float).reshape(1, 2))
    #print(str(index) + ' : ' + str(loss))
    index += 1
    render()
    #if(index % 100 == 0):
        #kinemitic_model.save_weights('model', True)