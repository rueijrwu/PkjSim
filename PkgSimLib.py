from ipywidgets import widgets
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Arc, Polygon
import numpy as np
from mpl_toolkits.axes_grid.inset_locator import inset_axes

CSS_STYLE = """
<style>
.pbody {
    line-height: 250%;
    text-align: justify;
    font-size: 20px;
}
.item {
    margin-left:24px;
}
</style>
"""

POS_CORNEA = 13.5
RAD_CORNEA = 7.8
CEN_CORNEA = POS_CORNEA - RAD_CORNEA

POS_LEN = POS_CORNEA - 7.6
RAD_LEN = 6
CEN_LEN = POS_LEN + RAD_LEN

IDX_EYE = 1.377
RAD_EYE = 13

TEHTA_MAX = np.pi / 2


class M_I1(object):

    def __init__(self):
        self.f = RAD_CORNEA * IDX_EYE / (IDX_EYE - 1)
        self.lenRadius = RAD_CORNEA
        self.lenCenter = np.array([CEN_CORNEA, 0])

        self.qVerts = []
        for theta in np.linspace(-TEHTA_MAX, TEHTA_MAX, num=100):
            o, n, qPt = self.calculateImage(theta)
            self.qVerts.append(qPt)
        self.qVerts = np.array(self.qVerts)

    def calculateImage(self, theta):

        COS = np.cos(theta)
        SIN = np.sin(theta)
        n = np.array([COS, SIN])
        o = self.lenRadius * n + self.lenCenter
        qPt = -self.f * n + self.lenCenter
        return o, n, qPt


class M_I2(object):

    def __init__(self, parent):
        self.f = RAD_LEN / 2
        self.lenRadius = RAD_LEN
        self.lenCenter = np.array([CEN_LEN, 0])
        self.parent = parent
        self.qVerts = []
        for theta in np.linspace(-TEHTA_MAX, TEHTA_MAX, num=100):
            o, n, pPt, qPt = self.calculateImage(theta)
            self.qVerts.append(qPt)
        self.qVerts = np.array(self.qVerts)

    def calculateImage(self, theta):

        o, n, pPt = self.parent.calculateImage(theta)
        n = pPt - self.lenCenter
        n = n / np.linalg.norm(n)
        o = self.lenRadius * n + self.lenCenter
        p = np.linalg.norm(pPt - o)
        q = self.f * p / (p - self.f)
        qPt = -q * n + o
        return o, n, pPt, qPt


class M_I3(object):

    def __init__(self, parent):
        self.f = RAD_CORNEA / (IDX_EYE - 1)
        self.lenRadius = RAD_CORNEA
        self.lenCenter = np.array([CEN_CORNEA, 0])
        self.parent = parent
        self.qVerts = []
        for theta in np.linspace(-TEHTA_MAX, TEHTA_MAX, num=100):
            o, n, pPt, qPt = self.calculateImage(theta)
            self.qVerts.append(qPt)
        self.qVerts = np.array(self.qVerts)

    def calculateImage(self, theta):

        o, n, qPt, pPt = self.parent.calculateImage(theta)
        n = pPt - self.lenCenter
        n = n / np.linalg.norm(n)
        o = self.lenRadius * n + self.lenCenter
        p = np.linalg.norm(pPt - o)
        q = -self.f * p / (self.f + p)
        qPt = q * n + o
        return o, n, pPt, qPt


class NormalLens:

    def __init__(self, n1, n2, R):

        f = np.abs(R * n2 / (n1 - n2))
        self.f = f
        XMAX = np.abs(3 * f)
        YMAX = np.abs(f)
        self.fig = plt.figure(figsize=(8, 4), facecolor='white')
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.axis('off')
        self.ax.axis([-XMAX, XMAX, -YMAX, YMAX])

        self.lenCenter = Circle([-R, 0], radius=0.4, zorder=0, fc='black')
        self.len = Arc([-R, 0], 2 * R, 2 * R, theta1=-80.,
                       theta2=80.0, zorder=0, fc='white', ec='black', lw=2)
        self.lenCenterText = self.ax.text(-R, -2, r"$c$", fontsize=24)

        self.haxis = Line2D([-XMAX, XMAX], [0, 0],
                            zorder=1, lw=2, color='black')

        p = np.abs(2 * f)
        q = self.f * p / (self.f - p)
        self.p = Circle([p, 0], zorder=2, radius=0.4, fc='r', ec='none')
        self.q = Circle([q, 0], zorder=2, radius=0.4, fc='b', ec='none')
        self.o = Circle([0, 0], zorder=2, radius=0.4, fc='white', ec='black')
        self.pText = self.ax.text(p, -2, r"$p$", fontsize=24)
        self.qText = self.ax.text(q, -2, r"$q$", fontsize=24)
        self.oText = self.ax.text(0, -2, r"$s$", fontsize=24)

        self.ax.add_artist(self.lenCenter)
        self.ax.add_artist(self.haxis)
        self.ax.add_artist(self.p)
        self.ax.add_artist(self.q)
        self.ax.add_artist(self.o)
        self.ax.add_artist(self.len)

        self.sd = widgets.FloatSlider(
            min=0,
            max=5 * XMAX,
            step=0.25,
            value=np.abs(p),
            description=r'$\overline{sp}$ :',)
        self.sd.observe(self.update, names='value')
        display(self.sd)

    def update(self, change):
        x = change['new']
        q = self.f * x / (self.f - x)
        self.p.center = [x, 0]
        self.q.center = [q, 0]
        self.pText.set_position([x, -5])
        self.qText.set_position([q, -5])
        self.fig.canvas.draw_idle()


class FlatCircle(Polygon):

    def __init__(self,
                 center, radius, THETA1, THETA2,
                 closed=True, **kwargs):

        self._center = center
        self._theta1 = THETA1
        self._theta2 = THETA2
        self.setRadius(radius)
        super(self.__class__, self).__init__(
            self._verts, closed=closed, **kwargs)

    def setRadius(self, R):
        theta = np.linspace(self._theta1, self._theta2)
        self._verts = np.zeros((50, 2), dtype=np.float)
        self._verts[:, 0] = R * np.cos(theta) + self._center[0]
        self._verts[:, 1] = R * np.sin(theta) + self._center[1]

    def updateVerts(self):
        self.set_xy(self._verts)


class Trace(Polygon):

    def __init__(self, verts, closed=True, **kwargs):
        super(self.__class__, self).__init__(verts, closed=closed, **kwargs)

    def updateVerts(self, verts):
        self.set_xy(verts)


class RotatedLens:

    def __init__(self, n1, n2, R):
        f = np.abs(R * n2 / (n1 - n2))

        self.f = f
        self.lenCenter = np.array([5, 0])
        self.lenRadius = R
        self.pRadius = np.abs(1.25 * f) + R + 5
        self.rotCenter = np.array([0, 0])
        self.theta = 0
        self.XMAX = XMAX = np.abs(6 * f)

        self.fig = plt.figure(figsize=(8, 6), facecolor='white')
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.axis('off')
        self.ax.axis([-XMAX, XMAX, -XMAX, XMAX])

        # Lens
        self.len = Arc(self.lenCenter, 2 * R, 2 * R, theta1=-80.,
                       theta2=80.0, zorder=0, fc='white', ec='black', lw=2)
        self.lenDot = Circle([0, 0], radius=0.6, zorder=2,
                             fc='white', ec='black')
        self.lenText = self.ax.text(0, -4, r"$s$", fontsize=20)

        # Rotation Center)
        self.rotCenterDot = Circle(
            self.rotCenter, radius=0.6, zorder=2, fc='g', ec='none')
        self.rotCenterText = self.ax.text(
            self.rotCenter[0] - 2, 1.5, r"$o$", fontsize=20)

        # Lens Center
        self.lenCenterDot = Circle(
            self.lenCenter, radius=0.4, zorder=2, fc='black')
        self.lenCenterText = self.ax.text(
            self.lenCenter[0] - 2, 1.5, r"$c$", fontsize=20)

        # Horizontal Axis
        self.haxis = Line2D([-XMAX, XMAX], [0, 0],
                            zorder=1, lw=2, ls='--', color='black')

        # Optical axis
        self.optAxis = Line2D([-XMAX, XMAX],
                              [0, 0],
                              zorder=1, lw=2, color='black')

        # Dot & Text for p
        self.pDot = Circle([0, 0], zorder=3, radius=0.6, fc='r', ec='none')
        self.pText = self.ax.text(0, -4, r"$p$", fontsize=20)
        self.pPath = FlatCircle(
            self.rotCenter,
            self.pRadius,
            -np.pi / 2, np.pi / 2,
            closed=False, fc="none", ec="r",
            ls="--")

        # DOt & Text for q
        self.qDot = Circle([0, 0], zorder=3, radius=0.6, fc='b', ec='none')
        self.qText = self.ax.text(0, -4, r"$q$", fontsize=20)
        verts = []
        for theta in np.arange(-np.pi / 3, np.pi / 3, np.pi * 2 / 150):
            o, n, pPt, qPt = self.calculateImage(theta)
            verts.append(qPt)
        self.qPath = Trace(verts, closed=False, fc="none", ec="b", ls="--")

        # P radius
        self.pRadiusLine = Line2D([0, 0],
                                  [0, 0],
                                  zorder=1, lw=2, color='purple', ls='--')

        # Add artist to plot
        self.ax.add_artist(self.haxis)
        self.ax.add_artist(self.len)
        self.ax.add_artist(self.lenDot)
        self.ax.add_artist(self.lenCenterDot)
        self.ax.add_artist(self.optAxis)
        self.ax.add_artist(self.pDot)
        self.ax.add_artist(self.pPath)
        self.ax.add_artist(self.pRadiusLine)
        self.ax.add_artist(self.qDot)
        self.ax.add_artist(self.qPath)
        self.ax.add_artist(self.rotCenterDot)
        self.update()

        # ===== Interactive widget ======
        # Slider for q
        self.tSd = widgets.FloatSlider(
            min=-60, max=60, step=1,
            value=0, description=r"$\theta$ :")
        self.tSd.observe(self.onThetaChange, names='value')

        p = self.pRadius - self.lenCenter[0]
        # Slider for p
        self.pSd = widgets.FloatSlider(
            min=0, max=2 * XMAX, step=.25,
            value=p, description=r"$\overline{sp}$ :")
        self.pSd.observe(self.onPChange, names='value')
        display(self.tSd)
        display(self.pSd)

    def onThetaChange(self, change):
        self.theta = np.deg2rad(change['new'])
        self.update()

    def onPChange(self, change):
        self.pRadius = change['new']
        self.pPath.setRadius(self.pRadius)
        self.pPath.updateVerts()
        verts = []
        for theta in np.arange(-np.pi / 3, np.pi / 3, np.pi * 2 / 150):
            o, n, pPt, qPt = self.calculateImage(theta)
            verts.append(qPt)
        self.qPath.updateVerts(verts)
        self.update()

    def calculateImage(self, theta):

        COS = np.cos(theta)
        SIN = np.sin(theta)
        pPt = self.pRadius * np.array([COS, SIN]) + self.rotCenter

        # calcuate the intercetion of optical axis and lens
        n = pPt - self.lenCenter
        n = n / np.linalg.norm(n)
        o = self.lenRadius * n + self.lenCenter
        p = np.linalg.norm(pPt - o)
        q = self.f * p / (self.f - p)
        qPt = q * n + o
        return o, n, pPt, qPt

    def update(self):
        o, n, pPt, qPt = self.calculateImage(self.theta)
        self.pDot.center = pPt
        self.qDot.center = qPt
        self.lenDot.center = o
        self.pRadiusLine.set_data([0, pPt[0]], [0, pPt[1]])

        self.pText.set_position([pPt[0], pPt[1] - 4])
        self.qText.set_position([qPt[0], qPt[1] - 4])
        self.lenText.set_position([o[0], o[1] - 4])

        p1 = 2 * self.XMAX * n + self.lenCenter
        p2 = - 2 * self.XMAX * n + self.lenCenter
        self.optAxis.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        self.fig.canvas.draw_idle()


# First Purkinje Image
class FirstPkg:

    def __init__(self):
        self.lenRadius = R = RAD_CORNEA
        self.f = self.lenRadius / 2
        self.lenCenter = np.array([CEN_CORNEA, 0])
        self.rotCenter = np.array([0, 0])
        self.theta = 0
        self.XMAX = XMAX = 3 * self.lenRadius

        self.fig = plt.figure(figsize=(8, 4), facecolor='white')
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.axis('off')
        self.ax.axis([-XMAX, XMAX, -XMAX, XMAX])

        #self.eye = FlatCircle(
        #    [0, 0],
        #    RAD_EYE,
        #    np.pi / 5, np.pi * 9 / 5,
        #    closed=False,
        #    fc='none')

        self.eye = Arc(
            [0, 0],
            2 * RAD_EYE, 2 * RAD_EYE,
            theta1=36, theta2=324,
            fc='none', lw=2)
        
        self.cornea = Arc(
            [CEN_CORNEA, 0],
            2 * RAD_CORNEA, 2 * RAD_CORNEA,
            theta1=-60., theta2=60.0,
            zorder=0, fc='white', ec='black', lw=2)
        
        
        # Lens
        self.len = Arc(self.lenCenter, 2 * R, 2 * R, theta1=-60.,
                       theta2=60.0, zorder=0, fc='white', ec='black', lw=2)
        self.lenDot = Circle([0, 0], radius=0.6, zorder=2,
                             fc='white', ec='black')
        self.lenText = self.ax.text(0, -4, r"$s$", fontsize=20)

        # Rotation Center)
        self.rotCenterDot = Circle(
            self.rotCenter, radius=0.6, zorder=2, fc='g', ec='none')
        self.rotCenterText = self.ax.text(
            self.rotCenter[0] - 2, 1.5, r"$o$", fontsize=20)

        # Lens Center
        self.lenCenterDot = Circle(
            self.lenCenter, radius=0.4, zorder=2, fc='black')
        self.lenCenterText = self.ax.text(
            self.lenCenter[0] - 2, 1.5, r"$c$", fontsize=20)

        # Horizontal Axis
        self.haxis = Line2D([-XMAX, XMAX], [0, 0],
                            zorder=1, lw=2, ls='--', color='black')

        # Optical axis
        self.optAxis = Line2D([-XMAX, XMAX],
                              [0, 0],
                              zorder=1, lw=2, color='black')

        # Optical axis
        self.lightBeam = Line2D([-XMAX, XMAX],
                                [0, 0],
                                zorder=2, lw=2, color='red')

        # DOt & Text for q
        self.qDot = Circle([0, 0], zorder=3, radius=0.5, fc='b', ec='none')
        self.qText = self.ax.text(0, -4, r"$q$", fontsize=20)
        verts = []
        for theta in np.linspace(-TEHTA_MAX, TEHTA_MAX, num=100):
            o, n, qPt = self.calculateImage(theta)
            verts.append(qPt)
        self.qPath = Trace(verts, closed=False, fc="none", ec="b", ls="--")

        # Add artist to plot
        self.ax.add_artist(self.eye)
        self.ax.add_artist(self.cornea)
        self.ax.add_artist(self.haxis)
        self.ax.add_artist(self.len)
        self.ax.add_artist(self.lenDot)
        self.ax.add_artist(self.lenCenterDot)
        self.ax.add_artist(self.optAxis)
        self.ax.add_artist(self.lightBeam)
        self.ax.add_artist(self.qDot)
        self.ax.add_artist(self.qPath)
        self.ax.add_artist(self.rotCenterDot)
        self.update()

        # ===== Interactive widget ======
        # Slider for q
        self.tSd = widgets.FloatSlider(
            min=-60, max=60, step=1,
            value=0, description=r"$\theta$ :")
        self.tSd.observe(self.onThetaChange, names='value')
        display(self.tSd)

    def onThetaChange(self, change):
        self.theta = np.deg2rad(change['new'])
        self.update()

    def calculateImage(self, theta):

        COS = np.cos(theta)
        SIN = np.sin(theta)
        n = np.array([COS, SIN])
        o = self.lenRadius * n + self.lenCenter
        qPt = self.f * n + self.lenCenter
        return o, n, qPt

    def update(self):
        o, n, qPt = self.calculateImage(self.theta)
        self.qDot.center = qPt
        self.lenDot.center = o
        # self.pRadiusLine.set_data([0, pPt[0]], [0, pPt[1]])
        # self.pText.set_position([pPt[0], pPt[1] - 4])
        self.qText.set_position([qPt[0], qPt[1] - 4])
        self.lenText.set_position([o[0], o[1] - 4])

        p1 = 2 * self.XMAX * n + self.lenCenter
        p2 = - 2 * self.XMAX * n + self.lenCenter
        self.optAxis.set_data([p1[0], p2[0]], [p1[1], p2[1]])

        COS = np.cos(self.theta)
        SIN = np.sin(self.theta)
        self.lightBeam.set_data([0, self.XMAX * COS], [0, self.XMAX * SIN])
        self.fig.canvas.draw_idle()

class ImgPlot(object):

    def __init__(self):
        self.rotCenter = np.array([0, 0])
        self.XMAX = 25

        self.fig = plt.figure(figsize=(8, 4), facecolor='white')
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.axis('off')
        self.ax.axis([-self.XMAX, self.XMAX, -self.XMAX, self.XMAX])

        self.eye = Arc(
            [0, 0],
            2 * RAD_EYE, 2 * RAD_EYE,
            theta1=36, theta2=324,
            fc='none')

        # CORNEA
        self.cornea = Arc(
            [CEN_CORNEA, 0],
            2 * RAD_CORNEA, 2 * RAD_CORNEA,
            theta1=-60., theta2=60.0,
            zorder=0, fc='white', ec='black', lw=2)

        # LEN
        self.LEN = Arc(
            [CEN_LEN, 0],
            2 * RAD_LEN, 2 * RAD_LEN,
            theta1=120., theta2=240.,
            zorder=0, fc='white', ec='black', lw=2)

        # Rotation Center)
        self.rotCenterDot = Circle(
            self.rotCenter, radius=0.6, zorder=2, fc='g', ec='none')
        self.rotCenterText = self.ax.text(
            self.rotCenter[0] - 2, 1.5, r"$o$", fontsize=20)

        # Horizontal Axis
        self.haxis = Line2D([-self.XMAX, self.XMAX], [0, 0],
                            zorder=1, lw=2, ls='--', color='black')

        # Optical axis
        self.optAxis = Line2D([-self.XMAX, self.XMAX],
                              [0, 0],
                              zorder=1, lw=2, color='black')

        # Lens
        self.lenDot = Circle([0, 0], radius=0.6, zorder=2,
                             fc='white', ec='black')
        self.lenText = self.ax.text(0, -4, r"$s$", fontsize=20)

        # DOt & Text for q
        self.qDot = Circle([0, 0], zorder=3, radius=0.5, fc='b', ec='none')

        self.lightBeam = Line2D([-self.XMAX, self.XMAX],
                                [0, 0],
                                zorder=2, lw=2, color='purple')

        self.ax.add_artist(self.eye)
        self.ax.add_artist(self.cornea)
        self.ax.add_artist(self.haxis)
        self.ax.add_artist(self.LEN)
        self.ax.add_artist(self.optAxis)
        self.ax.add_artist(self.rotCenterDot)
        self.ax.add_artist(self.lenDot)
        self.ax.add_artist(self.qDot)
        self.ax.add_artist(self.lightBeam)

        # ===== Interactive widget ======
        # Slider for q
        self.tSd = widgets.FloatSlider(
            min=-60, max=60, step=1,
            value=0, description=r"$\theta$ :")
        self.tSd.observe(self.onThetaChange, names='value')
        display(self.tSd)

    def onThetaChange(self, change):
        self.theta = np.deg2rad(change['new'])
        self.update()

# Plot for the image of object by conrea
class I1Plot(ImgPlot):

    def __init__(self):
        super(I1Plot, self).__init__()
        self.theta = 0
        self.model = M_I1()

        # Lens Center
        self.lenCenterDot = Circle(
            self.model.lenCenter, radius=0.4, zorder=2, fc='black')
        self.lenCenterText = self.ax.text(
            self.model.lenCenter[0] - 2, 1.5, r"$c$", fontsize=20)

        self.qText = self.ax.text(0, -4, r"$I_1$", color="b", fontsize=20)
        self.qPath = Trace(self.model.qVerts, closed=False,
                           fc="none", ec="b", ls="--")

        # Add artist to plot
        self.ax.add_artist(self.lenCenterDot)
        self.ax.add_artist(self.qPath)
        self.ax.add_artist(self.rotCenterDot)
        self.update()

    def update(self):
        o, n, qPt = self.model.calculateImage(self.theta)
        self.qDot.center = qPt
        self.lenDot.center = o
        self.qText.set_position([qPt[0], qPt[1] - 4])
        self.lenText.set_position([o[0], o[1] - 4])

        p1 = 2 * self.XMAX * n + self.model.lenCenter
        p2 = - 2 * self.XMAX * n + self.model.lenCenter
        self.optAxis.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        COS = np.cos(self.theta)
        SIN = np.sin(self.theta)
        self.lightBeam.set_data([0, self.XMAX * COS], [0, self.XMAX * SIN])
        self.fig.canvas.draw_idle()

    def getModel(self):
        return self.model


class I2Plot(ImgPlot):

    def __init__(self, parent):
        super(I2Plot, self).__init__()
        self.theta = 0
        self.model = M_I2(parent)

        # Lens Center
        self.lenCenterDot = Circle(
            self.model.lenCenter, radius=0.4, zorder=2, fc='black')
        self.lenCenterText = self.ax.text(
            self.model.lenCenter[0] - 2, 1.5, r"$c$", fontsize=20)

        # Dot & Text for p
        self.pDot = Circle([0, 0], zorder=3, radius=0.5, fc='r', ec='none')
        self.pText = self.ax.text(0, -4, r"$I_1$", color="r", fontsize=20)
        self.pPath = Polygon(
            self.model.parent.qVerts,
            closed=False, fc="none", ec="r",
            ls="--")

        # DOt & Text for q
        self.qText = self.ax.text(0, -4, r"$I_2$", color="b", fontsize=20)
        self.qPath = Trace(self.model.qVerts, closed=False,
                           fc="none", ec="b", ls="--")

        # Add artist to plot
        self.ax.add_artist(self.lenCenterDot)
        self.ax.add_artist(self.pDot)
        self.ax.add_artist(self.pPath)
        self.ax.add_artist(self.qPath)
        self.update()

    def update(self):
        o, n, pPt, qPt = self.model.calculateImage(self.theta)
        self.pDot.center = pPt
        self.qDot.center = qPt
        self.lenDot.center = o
        self.pText.set_position([pPt[0], pPt[1] - 4])
        self.qText.set_position([qPt[0], qPt[1] - 4])
        self.lenText.set_position([o[0], o[1] - 4])

        p1 = 2 * self.XMAX * n + self.model.lenCenter
        p2 = - 2 * self.XMAX * n + self.model.lenCenter
        self.optAxis.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        COS = np.cos(self.theta)
        SIN = np.sin(self.theta)
        self.lightBeam.set_data([0, self.XMAX * COS], [0, self.XMAX * SIN])
        self.fig.canvas.draw_idle()

    def getModel(self):
        return self.model


class I3Plot(ImgPlot):

    def __init__(self, parent):
        super(I3Plot, self).__init__()
        self.theta = 0
        self.model = M_I3(parent)

        # Lens Center
        self.lenCenterDot = Circle(
            self.model.lenCenter, radius=0.4, zorder=2, fc='black')
        self.lenCenterText = self.ax.text(
            self.model.lenCenter[0] - 2, 1.5, r"$c$", fontsize=20)

        # Dot & Text for p
        self.pDot = Circle([0, 0], zorder=3, radius=0.3, fc='r', ec='none')
        self.pText = self.ax.text(0, -2, r"$I_2$", color="r", fontsize=20)
        self.pPath = Polygon(
            self.model.parent.qVerts,
            closed=False, fc="none", ec="r",
            ls="--")

        # DOt & Text for q
        self.qText = self.ax.text(0, 2, r"$q$", color="b", fontsize=20)
        self.qPath = Trace(self.model.qVerts, closed=False,
                           fc="none", ec="b", ls="--")
        self.qDot.set_radius(0.3)
        self.lenCenterDot.set_radius(0.3)
        self.lenDot.set_radius(0.3)
        self.XMAX = self.XMAX / 3
        self.ax.axis([self.XMAX / 3, 3 * self.XMAX, -self.XMAX, self.XMAX])

        # Add artist to plot
        self.ax.add_artist(self.lenCenterDot)
        self.ax.add_artist(self.pDot)
        self.ax.add_artist(self.pPath)
        self.ax.add_artist(self.qPath)
        self.update()

    def update(self):
        o, n, pPt, qPt = self.model.calculateImage(self.theta)
        self.pDot.center = pPt
        self.qDot.center = qPt
        self.lenDot.center = o
        self.pText.set_position([pPt[0], pPt[1] + 1.5])
        self.qText.set_position([qPt[0], qPt[1] - 1.5])
        self.lenText.set_position([o[0], o[1] - 1.5])

        p1 = 2 * self.XMAX * n + self.model.lenCenter
        p2 = - 2 * self.XMAX * n + self.model.lenCenter
        self.optAxis.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        COS = np.cos(self.theta)
        SIN = np.sin(self.theta)
        self.lightBeam.set_data(
            [0, 3 * self.XMAX * COS], [0, 3 * self.XMAX * SIN])
        self.fig.canvas.draw_idle()


class PkgPlot(ImgPlot):

    def __init__(self, parent):
        super(PkgPlot, self).__init__()
        #self.fig1 = plt.figure()

        self.theta = 0
        self.model = M_I3(parent)

        self.XMAX = self.XMAX / 3
        self.ax.axis([self.XMAX / 3, 3 * self.XMAX, -self.XMAX, self.XMAX])

        self.lenDot.remove()
        self.rotCenterDot.remove()
        self.rotCenterText.remove()
        self.lenText.remove()

        # Dot & Text for p
        self.pDot = Circle([0, 0], zorder=3, radius=0.2, fc='r', ec='none')
        self.pText = self.ax.text(0, -1.5, r"$P_1$", color="r", fontsize=16)

        self.f1 = RAD_CORNEA / 2
        self.c1 = np.array([CEN_CORNEA, 0])
        theta = np.linspace(-TEHTA_MAX, TEHTA_MAX, num=100)
        verts = self.f1 * \
            np.array([np.cos(theta), np.sin(theta)]).T + self.c1.reshape(1, 2)
        self.pPath = Polygon(
            verts,
            closed=False, fc="none", ec="r")

        # DOt & Text for q
        self.qDot.set_radius(0.2)
        self.qText = self.ax.text(0, 1.5, r"$P_4$", color="b", fontsize=16)
        self.qPath = Trace(self.model.qVerts, closed=False,
                           fc="none", ec="b")

        self.cameraPlane = Line2D(
            [POS_CORNEA, POS_CORNEA], [-10, 10],
            zorder=1, lw=6, color='gray')
        self.pDotProj = Circle(
            [0, 0], zorder=3, radius=0.2, fc='white', ec='r', lw=2)
        self.qDotProj = Circle(
            [0, 0], zorder=3, radius=0.2, fc='white', ec='b', lw=2)
        self.pProj = Line2D(
            [0, 0], [0, 0],
            zorder=1, lw=2, color='r', ls="--")
        self.qProj = Line2D(
            [0, 0], [0, 0],
            zorder=1, lw=2, color='b', ls="--")

        # Add artist to plot
        self.ax.add_artist(self.pDot)
        self.ax.add_artist(self.pPath)
        self.ax.add_artist(self.qPath)
        self.ax.add_artist(self.cameraPlane)
        self.ax.add_artist(self.pDotProj)
        self.ax.add_artist(self.qDotProj)
        self.ax.add_artist(self.pProj)
        self.ax.add_artist(self.qProj)

        self.ax1 = inset_axes(self.ax,
                              width=3.,  # width = 30% of parent_bbox
                              height=2.1,  # height : 1 inch
                              bbox_to_anchor=(1, 0.95),
                              bbox_transform=self.ax.figure.transFigure)
        n1 = np.array([np.sin(theta), -np.cos(theta)]).T
        sep = self.model.qVerts - verts
        trace = []
        for n, s in zip(n1, sep):
            trace.append(np.dot(n, s))
        trace = np.array(trace)
        AMP = (trace.max() - trace.min()) / 2
        AVG = (trace.max() + trace.min()) / 2
        Theta = theta / np.pi * 180
        self.ax1.plot(Theta, trace)
        self.ax1.plot(
            Theta,
            AMP * np.sin(theta) + AVG,
            'r--')
        self.sepDot, = self.ax1.plot([0], [0], 'ob')
        self.ax1.set_ylabel("mm")
        self.ax1.set_xlabel("deg")
        self.ax1.set_title("Seperation of Purkinje image")
        self.ax1.legend(["Seperation", "Sin"], bbox_to_anchor=(0.45, 1))

        self.update()

    def update(self):
        o, n, pPt, qPt = self.model.calculateImage(self.theta)
        COS = np.cos(self.theta)
        SIN = np.sin(self.theta)
        n1 = np.array([COS, SIN])
        pPt = self.f1 * n1 + self.c1
        self.pDot.center = pPt
        self.qDot.center = qPt
        self.pText.set_position([pPt[0], pPt[1] - 1.5])
        self.qText.set_position([qPt[0], qPt[1] + 1.5])

        n2 = np.array([SIN, -COS])
        lc = RAD_CORNEA * n1 + [CEN_CORNEA, 0]
        p1 = lc + 10 * n2
        p2 = lc - 10 * n2
        self.sepDot.set_data(np.rad2deg(self.theta),
                             np.dot(qPt - pPt, n2))
        self.cameraPlane.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        p1 = np.dot((pPt - lc), n2) * n2 + lc
        p2 = np.dot((qPt - lc), n2) * n2 + lc
        self.pDotProj.center = p1
        self.qDotProj.center = p2
        self.pProj.set_data([pPt[0], p1[0]], [pPt[1], p1[1]])
        self.qProj.set_data([qPt[0], p2[0]], [qPt[1], p2[1]])
        self.lightBeam.set_data(
            [0, 3 * self.XMAX * COS], [0, 3 * self.XMAX * SIN])
        self.fig.canvas.draw_idle()
        # self.fig1.canvas.draw_idle()
