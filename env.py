import cv2
from collections import namedtuple
import numpy as np

Point = namedtuple('Point', ['x', 'y'])


class Env():
    def __init__(self,
                 model_type=0,
                 speed=0.01,
                 angle_range0=(-0.25 * np.pi, 0.25 * np.pi),
                 angle_range1=(-0.5 * np.pi, 0),
                 image_size=(300, 350),
                 model = None):
        self.model_type = model_type
        self.speed = speed
        self.angle_range0 = angle_range0
        self.angle_range1 = angle_range1
        self.image_size = image_size
        self.model = model
        self.min_dist = 10
        self.target_point = Point(0, 0)
        self.l0 = 70
        self.l1 = 60
        self.l2 = 50
        self.theta0 = 0
        self.theta1 = 0
        self.theta2 = 0
        self.point0 = Point(80, self.image_size[1] // 2)
        self.n_action = 2
        self.reset()

    def reset(self, reset_rotation=True):
        self.target_point = self.random_point()
        if reset_rotation:
            self.theta0 = np.random.uniform(*self.angle_range0)
            self.theta1 = np.random.uniform(*self.angle_range1)
            self.theta2 = self.theta1
        self._update_point()
        s = self.get_state()
        return s

    def step(self, action):
        r = 0
        if self.model_type == 0:
            self.theta0 = self.angle_range0[0] + (self.angle_range0[1] - self.angle_range0[0]) * action[0]
            self.theta1 = self.angle_range1[0] + (self.angle_range1[1] - self.angle_range1[0]) * action[1]
            self.theta2 = self.theta1
        elif self.model_type == 1:
            self.theta0 += self.speed * action[0]
            if self.theta0 < self.angle_range0[0]:
                r -= 1
            if self.theta0 > self.angle_range0[1]:
                r -= 1
            self.theta0 = np.clip(self.theta0, self.angle_range0[0], self.angle_range0[1])
            self.theta1 += self.speed * action[1]
            if self.theta1 < self.angle_range1[0]:
                r -= 1
            if self.theta1 > self.angle_range1[1]:
                r -= 1
            self.theta1 = np.clip(self.theta1, self.angle_range1[0], self.angle_range1[1])
            self.theta2 = self.theta1
        else:
            return [], 0, True
        self._update_point()
        dist = np.sqrt((self.target_point.x - self.point3.x) ** 2 + (self.target_point.y - self.point3.y) ** 2)
        done = False
        if dist < self.min_dist:
            r += 1
            done = True
        else:
            r = -dist / self.image_size[0]
        s = self.get_state()
        return s, r, done

    def _update_point(self):
        x1 = self.point0.x + self.l0 * np.cos(self.theta0)
        y1 = self.point0.y - self.l0 * np.sin(self.theta0)
        self.point1 = Point(int(x1), int(y1))

        x2 = self.point1.x + self.l1 * np.cos(self.theta1 + self.theta0)
        y2 = self.point1.y - self.l1 * np.sin(self.theta1 + self.theta0)
        self.point2 = Point(int(x2), int(y2))

        x3 = self.point2.x + self.l2 * np.cos(self.theta2 + self.theta1 + self.theta0)
        y3 = self.point2.y - self.l2 * np.sin(self.theta2 + self.theta1 + self.theta0)
        self.point3 = Point(int(x3), int(y3))

    def random_point(self, eps=0):
        if self.model_type == 0:
            return Point(np.random.randint(0, self.image_size[0]), np.random.randint(0, self.image_size[1]))

        theta0 = np.random.uniform(self.angle_range0[0] - eps, self.angle_range0[1] + eps)
        theta1 = np.random.uniform(self.angle_range1[0] - eps, self.angle_range1[1] + eps)
        theta2 = theta1
        x1 = self.point0.x + self.l0 * np.cos(theta0)
        y1 = self.point0.y - self.l0 * np.sin(theta0)
        point1 = Point(x1, y1)

        x2 = point1.x + self.l1 * np.cos(theta1 + theta0)
        y2 = point1.y - self.l1 * np.sin(theta1 + theta0)
        point2 = Point(x2, y2)

        x3 = point2.x + self.l2 * np.cos(theta2 + theta1 + theta0)
        y3 = point2.y - self.l2 * np.sin(theta2 + theta1 + theta0)
        point3 = Point(int(x3), int(y3))
        return point3

    def render(self, frame_time=1):
        img = np.zeros((self.image_size[1], self.image_size[0], 3), np.uint8)
        img.fill(255)

        cv2.line(img, self.point0, self.point1, (150, 30, 30), 4, lineType=cv2.LINE_AA)
        cv2.line(img, self.point1, self.point2, (150, 30, 30), 3, lineType=cv2.LINE_AA)
        cv2.line(img, self.point2, self.point3, (150, 30, 30), 2, lineType=cv2.LINE_AA)
        cv2.circle(img, self.target_point, int(self.min_dist), (75, 75, 75), -1, lineType=cv2.LINE_AA)
        cv2.imshow('screen', img)
        cv2.waitKey(frame_time)

    def get_state(self):
        x1 = self.point1.x / self.image_size[0]
        y1 = self.point1.y / self.image_size[1]
        x2 = self.point2.x / self.image_size[0]
        y2 = self.point2.y / self.image_size[1]
        x3 = self.point3.x / self.image_size[0]
        y3 = self.point3.y / self.image_size[1]
        xt = self.target_point.x / self.image_size[0]
        yt = self.target_point.y / self.image_size[1]
        return np.array([x1, y1, x2, y2, x3, y3, xt, yt])
