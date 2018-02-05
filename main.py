from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-title TrenchAssessment')
loadPrcFileData("", "win-size 1280 960")
loadPrcFileData('', 'sync-video false')
loadPrcFileData('', 'show-frame-rate-meter true')
loadPrcFileData('', 'texture-minfilter linear-mipmap-linear')

import random
import math
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectButton import DirectButton
from panda3d.core import PerspectiveLens, TextNode, \
TexGenAttrib, TextureStage, TransparencyAttrib, LPoint3, VBase4, Texture, BitMask32, CompassEffect, WindowProperties, \
DirectionalLight, AmbientLight, Point3D, deg2Rad, NodePath, GeomNode, GeomVertexReader
from panda3d.egg import EggData, EggVertex, EggVertexPool, EggPolygon, loadEggData

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D

from cameracontroller import FirstPersonCamera
from threeaxisgrid import ThreeAxisGrid
from picker import Picker

''' Constants '''
WHITE = (1, 1, 1, 1)
# Distance to which vertices will be included in plane consideration
PROFILE_DELTA = 0.03

def add_instructions(pos, msg):
	"""Function to put instructions on the screen."""
	return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1),
						parent=base.a2dTopLeft, align=TextNode.ALeft,
						pos=(0.08, -pos - 0.04), scale=.05)

def add_title(text):
	"""Function to put title on the screen."""
	return OnscreenText(text=text, style=1, pos=(-0.1, 0.09), scale=.08,
						parent=base.a2dBottomRight, align=TextNode.ARight,
						fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1))

class Game(ShowBase):
	"""Sets up the game, camera, controls, and loads models."""
	def __init__(self):
		ShowBase.__init__(self)
		self.xray_mode = False
		self.show_model_bounds = False

		self.lastHighlight = None

		# Display instructions
		add_title("TrenchAssessment")
		add_instructions(0.06, "[Esc]: Quit")
		#add_instructions(0.12, "[W]: Move Forward")
		#add_instructions(0.18, "[A]: Move Left")
		#add_instructions(0.24, "[S]: Move Right")
		#add_instructions(0.30, "[D]: Move Back")
		#add_instructions(0.36, "Arrow Keys: Look Around")
		add_instructions(0.12, "[F]: Toggle Wireframe")
		add_instructions(0.18, "[X]: Toggle X-Ray Mode")
		add_instructions(0.24, "[B]: Toggle Bounding Volumes")

		self.keys = {}
		self.accept('f', self.toggleWireframe)
		self.accept('x', self.toggle_xray_mode)
		self.accept('b', self.toggle_model_bounds)
		self.accept('escape', __import__('sys').exit, [0])
		self.disableMouse()

		# Setup camera
		self.cameraarm = render.attachNewNode('arm')
		self.cameraarm.setEffect(CompassEffect.make(render)) # NOT inherit rotation
		self.lens = PerspectiveLens()
		self.lens.setFov(60)
		self.lens.setNear(0.01)
		self.lens.setFar(10000000.0)
		self.cam.node().setLens(self.lens)
		self.heading = 0
		self.pitch = 0.0
		self.camera.reparentTo(self.cameraarm)
		self.camera.setY(-10) # camera distance from model
		self.accept('wheel_up', lambda : base.camera.setY(self.camera.getY()+400 * globalClock.getDt()))
		self.accept('wheel_down', lambda : base.camera.setY(self.camera.getY()-400 * globalClock.getDt()))
		self.accept('shift-wheel_up', lambda : base.camera.setY(self.camera.getY()+2000 * globalClock.getDt()))
		self.accept('shift-wheel_down', lambda : base.camera.setY(self.camera.getY()-2000 * globalClock.getDt()))
		self.isMouse3Down = False
		self.accept('mouse3', self.setMouse3Down)
		self.accept('mouse3-up', self.setMouse3Up)
		self.taskMgr.add(self.thirdPersonCameraTask, 'thirdPersonCameraTask')
		# Add Camera picking
		#self.rollover = Picker(render)
		#taskMgr.add(self.rolloverTask, 'rollover')
		self.picker = Picker(render)
		self.accept('mouse1', self.tryPick)
		self.accept('mouse1-up', self.stopRotation)

		# Setup Lights
		dlight = DirectionalLight('dlight')
		dlight2 = DirectionalLight('dlight2')
		alight = AmbientLight('alight')
		alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
		self.lightnode = render.attachNewNode(dlight)
		self.lightnode2 = render.attachNewNode(dlight2)
		self.lightnode2.setHpr(0, -90, 0)
		self.alightnode = render.attachNewNode(alight)
		render.setLight(self.lightnode)
		render.setLight(self.lightnode2)
		render.setLight(self.alightnode)

		# load models
		m = loader.loadModel("models/grube_simplified_rotated.egg")
		self.trench = render.attachNewNode("trench")
		m.reparentTo(self.trench)
		tbounds = self.trench.getTightBounds()
		p1 = tbounds[0]
		p2 = tbounds[1]
		po = p2-(p2-p1)/2
		m.setPos(-po)
		self.trench.setColor(WHITE)
		self.trench.setCollideMask(BitMask32.allOff())

		self.attachGizmos(self.trench)
		self.rotEnabled = False

		# load grid
		grid = ThreeAxisGrid()
		gridnodepath = grid.create()
		gridnodepath.setTransparency(TransparencyAttrib.MAlpha)
		gridnodepath.setAlphaScale(0.25)
		gridnodepath.reparentTo(render)

		# buttons
		a=DirectButton(text = ("X-Sections", "X-Sections", "X-Sections", "disabled"), scale=.05, pos=(-1.2,0,-0.95), command=self.calculateXSessions)
		a=DirectButton(text = ("RotGimbal", "RotGimbal", "RotGimbal", "disabled"), scale=.05, pos=(-0.9,0,-0.95), command=self.toggleRotGimbal)

	def calculateXSessions(self):
		"""
		Calculates cross-section profiles
		"""
		# Check for extension
		planes = self.createCuttingPlanes()
		profiles = self.calculateProfiles()

	def calculateProfiles(self):
		vertexToPlane = {}

		# init dict
		for p in self.planedata:
			vertexToPlane[p] = [[],[]]

		geomNodeCollection = self.trench.findAllMatches('**/+GeomNode')
		print(geomNodeCollection)
		# 0 will be the trench - change later
		geomNode = geomNodeCollection[0].node()
		for i in range(geomNode.getNumGeoms()):
			geom = geomNode.getGeom(i)
			state = geomNode.getGeomState(i)
			vdata = geom.getVertexData()
			vertex = GeomVertexReader(vdata, 'vertex')
			while not vertex.isAtEnd():
				v = vertex.getData3f()
				# calculate distance to planes
				for p in self.planedata:
					if self.deltaInclude(v, p):
						if v[1] not in vertexToPlane[p][0]: # prevent double addition of x values
							vertexToPlane[p][0].append(v[1])
							vertexToPlane[p][1].append(v[2])
		#print(vertexToPlane[1])




		# filter xy coordinates
		#new_x, new_y = self.deltaFilter()

		i = 0
		for p in self.planedata:
			plt.figure(i)

			# sort x coordinates
			L = sorted(zip(vertexToPlane[p][0], vertexToPlane[p][1]))
			new_x, new_y = zip(*L)

			# signal process y values
			yhat = savgol_filter(new_y, 21, 5) #81,5

			# 1d interpolation
			f = interp1d(new_x, yhat, 'quadratic')

			i += 1
			#print(new_x)
			plt.plot(new_x, new_y, color='k', alpha=0.5, linewidth=0.8)
			plt.plot(new_x, yhat, color='b', alpha=0.5, linewidth=0.8)
			plt.plot(new_x, f(new_x), color='r', alpha=0.5, linewidth=2.0)
			plt.legend(['original', 'savgol_filter', 'interpolated (cubic)'])
			#plt.plot(new_x, new_y)

			#plt.plot(new_x, f(new_x))
		plt.show()



	def deltaInclude(self, v, p):
		if v[0] >= p - PROFILE_DELTA and v[0] <= p + PROFILE_DELTA:
			return True
		else:
			return False

	def deltaFilter(self, v, p):
		return False

	def createCuttingPlanes(self):
		self.planes = self.render.attachNewNode("planes")
		pl = self.planes.attachNewNode("pl");
		data = EggData()
		self.planedata = []

		vp = EggVertexPool('plane')
		data.addChild(vp)

		fac = 1.0
		expanse = 10.0
		rng = 4
		for i in range(-rng,rng):
			poly = EggPolygon()
			data.addChild(poly)

			self.planedata.append(i*fac)

			v = EggVertex()
			v.setPos(Point3D(i*fac, -expanse, -expanse))
			poly.addVertex(vp.addVertex(v))
			v = EggVertex()
			v.setPos(Point3D(i*fac, -expanse, +expanse))
			poly.addVertex(vp.addVertex(v))
			v = EggVertex()
			v.setPos(Point3D(i*fac, +expanse, +expanse))
			poly.addVertex(vp.addVertex(v))
			v = EggVertex()
			v.setPos(Point3D(i*fac, +expanse, -expanse))
			poly.addVertex(vp.addVertex(v))

		node = loadEggData(data)
		np = NodePath(node)
		np.reparentTo(pl)
		np.setColor(0, 1, 0, 1)
		np.setTwoSided(True)
		np.setTransparency(TransparencyAttrib.MAlpha)
		np.setAlphaScale(0.1)
		np.setCollideMask(BitMask32(0x0))
		return self.planes

	def toggleRotGimbal(self):
		if self.rotEnabled:
			self.gizmos.setAlphaScale(0)
			self.rotEnabled = False
		else:
			self.gizmos.setAlphaScale(0.1)
			self.rotEnabled = True

	def startRotation(self):
		self.isRotating = True
		md = self.win.getPointer(0)
		self.mouselasttick = (md.getX(), md.getY())
		self.taskMgr.add(self.onRotateMoveTask, 'RotationTask')

	def stopRotation(self):
		self.isRotating = False
		self.taskMgr.remove("RotationTask")
		props = WindowProperties()
		props.setCursorHidden(False)
		self.win.requestProperties(props)
		#unselect on mouse up
		if self.lastHighlight:
			self.lastHighlight.setAlphaScale(0.1)
			self.lastHighlight = None

	def onRotateMoveTask(self, task):
		md = self.win.getPointer(0)
		props = WindowProperties()

		x = md.getX()
		y = md.getY()

		if self.isRotating:
			props.setCursorHidden(True)
			obj = self.trench
			rotAx = self.lastHighlight.getTag("axis")
			hpr = obj.getHpr(obj)
			if rotAx == "xy":
				hpr[0] = hpr[0] - (x - float(self.mouselasttick[0])) * 0.5
			elif rotAx == "xz":
				hpr[2] = hpr[2] - (y - float(self.mouselasttick[1])) * 0.5
			elif rotAx == "yz":
				hpr[1] = hpr[1] - (y - float(self.mouselasttick[1])) * 0.5

			obj.setHpr(obj, hpr)
			self.mouselasttick = (md.getX(), md.getY())
		else:
		   props.setCursorHidden(False)

		self.win.requestProperties(props)
		return task.cont

	def tryPick(self):
		obj, point, raw = self.picker.pick()
		#print(obj)
		if obj and self.rotEnabled:
			#dt = globalClock.getDt()
			#print(self.lastHighlight)
			if obj != self.lastHighlight:
				if self.lastHighlight:
					self.lastHighlight.setAlphaScale(0.1)
				#print("Set Highlight")
				self.lastHighlight = obj
				obj.setAlphaScale(0.3)
				self.startRotation()
		else:
			if self.lastHighlight != None:
				#print("Unset Highlight")
				self.lastHighlight.setAlphaScale(0.1)
				self.lastHighlight = None

	def rolloverTask(self, task):
		"""
		an alternate way to use the picker.
		Handle the mouse continuously
		"""
		obj, point, raw = self.rollover.pick()

		if obj:
			dt = globalClock.getDt()
			print(self.lastHighlight)
			if obj != self.lastHighlight or self.lastHighlight == None:
				print("Set Highlight")
				self.lastHighlight = obj
				obj.setAlphaScale(0.7)
		else:
			if self.lastHighlight:
				print("Unset Highlight")
				self.lastHighlight.setAlphaScale(0.1)
				self.lastHighlight = None

		return task.cont

	def attachGizmos(self, ref):
		self.gizmos = ref.attachNewNode("gizmos")
		self.gizmos.setAlphaScale(0)
		""" XZ """
		xz = self.gizmos.attachNewNode("xz");
		xzG = self.createGizmo(360, 30, 0, 10)
		xzG.setTransparency(TransparencyAttrib.MAlpha)
		xzG.setColor(0, 1, 0, 1)
		xzG.setTwoSided(True)
		xzG.reparentTo(xz)
		#xzG.setCollideMask(BitMask32(0x10))
		#xz.setPos(po)
		#xz.setAlphaScale(0.1)
		xz.setTag("axis", "xz")

		""" XY """
		xy = self.gizmos.attachNewNode("xy");
		xyG = self.createGizmo(360, 30, 1, 10)
		xyG.setTransparency(TransparencyAttrib.MAlpha)
		xyG.setColor(0, 0, 1, 1)
		xyG.setTwoSided(True)
		xyG.reparentTo(xy)
		#xzG.setCollideMask(BitMask32(0x10))
		#xy.setPos(po)
		#xy.setAlphaScale(0.1)
		xy.setTag("axis", "xy")

		""" YZ """
		yz = self.gizmos.attachNewNode("yz");
		yzG = self.createGizmo(360, 30, 2, 10)
		yzG.setTransparency(TransparencyAttrib.MAlpha)
		yzG.setColor(1, 0, 0, 1)
		yzG.setTwoSided(True)
		yzG.reparentTo(yz)
		#yz.setPos(po)
		#yz.setAlphaScale(0.1)
		yz.setTag("axis", "yz")

	def createGizmo(self, angleDegrees = 360, numSteps = 16, axis = 0, scale = 10):
		data = EggData()

		vp = EggVertexPool('fan')
		data.addChild(vp)

		poly = EggPolygon()
		data.addChild(poly)

		v = EggVertex()
		v.setPos(Point3D(0, 0, 0))
		poly.addVertex(vp.addVertex(v))

		angleRadians = deg2Rad(angleDegrees)

		for i in range(numSteps + 1):
			a = angleRadians * i / numSteps
			y = math.sin(a) * scale
			x = math.cos(a) * scale
			v = EggVertex()
			if axis is 0:
				v.setPos(Point3D(x, 0, y))
			elif axis is 1:
				v.setPos(Point3D(x, y, 0))
			else:
				v.setPos(Point3D(0, x, y))
			poly.addVertex(vp.addVertex(v))
		node = loadEggData(data)
		return NodePath(node)

	def setMouse3Down(self):
		md = self.win.getPointer(0)
		self.mouseorigin = (md.getX(), md.getY())
		self.mouselasttick = (md.getX(), md.getY())
		self.isMouse3Down = True
	def setMouse3Up(self):
		self.isMouse3Down = False
		self.win.movePointer(0, int(self.mouseorigin[0]), int(self.mouseorigin[1]))

	# camera rotation task
	def thirdPersonCameraTask(self, task):
	   md = self.win.getPointer(0)
	   props = WindowProperties()

	   x = md.getX()
	   y = md.getY()

	   if self.isMouse3Down:
		   props.setCursorHidden(True)
		   self.heading = self.heading - (x - float(self.mouselasttick[0])) * 0.5
		   self.pitch = self.pitch - (y - float(self.mouselasttick[1])) * 0.5
		   self.cameraarm.setHpr(self.heading, self.pitch,0)
		   self.mouselasttick = (md.getX(), md.getY())
	   else:
		   props.setCursorHidden(False)

	   self.win.requestProperties(props)
	   return task.cont
		#self.mouseLook = FirstPersonCamera(self, self.cam, self.render)

	def toggle_model_bounds(self):
		"""Toggle bounding volumes on and off on the models."""
		self.show_model_bounds = not self.show_model_bounds
		if self.show_model_bounds:
			self.trench.showBounds()
		else:
			self.trench.hideBounds()

	def toggle_xray_mode(self):
		"""Toggle X-ray mode on and off. This is useful for seeing the
		effectiveness of the occluder culling."""
		self.xray_mode = not self.xray_mode
		if self.xray_mode:
			self.trench.setColorScale((1, 1, 1, 0.5))
			self.trench.setTransparency(TransparencyAttrib.MDual)
		else:
			self.trench.setColorScaleOff()
			self.trench.setTransparency(TransparencyAttrib.MNone)

	def push_key(self, key, value):
		"""Stores a value associated with a key."""
		self.keys[key] = value

	def update(self, task):
		"""Updates the camera based on the keyboard input."""
		delta = globalClock.getDt()
		move_x = delta * 3 * -self.keys['a'] + delta * 3 * self.keys['d']
		move_z = delta * 3 * self.keys['s'] + delta * 3 * -self.keys['w']
		self.camera.setPos(self.camera, move_x, -move_z, 0)
		self.heading += (delta * 90 * self.keys['arrow_left'] +
						 delta * 90 * -self.keys['arrow_right'])
		self.pitch += (delta * 90 * self.keys['arrow_up'] +
					   delta * 90 * -self.keys['arrow_down'])
		self.camera.setHpr(self.heading, self.pitch, 0)
		return task.cont


game = Game()
game.run()
