#!/usr/bin/env python

import Box2D.b2 as b2
import math
import pygame
from pygame.locals import *  # for K_RIGHT, K_LEFT, etc.

class CartPole:
    def __init__(self):
        # Cart-pole physical parameters
        self.trackWidth = 5.0
        self.cartWidth = 0.3
        self.cartHeight = 0.2
        self.cartMass = 0.5
        self.poleMass = 0.1
        self.force = 0.15
        self.trackThickness = self.cartHeight
        self.poleLength = self.cartHeight * 6
        self.poleThickness = 0.04

        # Simulation parameters
        self.screenSize = (640, 480)  # Origin at the upper left
        self.worldSize = (float(self.trackWidth), float(self.trackWidth))  # Origin at center

        # Initialize the Box2D world
        self.world = b2.world(gravity=(0, -10), doSleep=True)
        self.framesPerSecond = 30  # Update rate
        self.velocityIterations = 8
        self.positionIterations = 6

        # Track color and pole collision category
        self.trackColor = (100, 100, 100)
        self.poleCategory = 0x0002

        # Initialize the cart and pole
        self._initialize_cart_pole()

    def _initialize_cart_pole(self):
        # This method should create the Box2D bodies for the cart and pole and set initial conditions
        self.cart = self.world.CreateDynamicBody(
            position=(0, self.cartHeight / 2),
            fixtures=b2.fixtureDef(
                shape=b2.polygonShape(box=(self.cartWidth / 2, self.cartHeight / 2)),
                density=self.cartMass,
            ),
        )
        
        self.pole = self.world.CreateDynamicBody(
            position=(0, self.cartHeight + self.poleLength / 2),
            fixtures=b2.fixtureDef(
                shape=b2.polygonShape(box=(self.poleThickness / 2, self.poleLength / 2)),
                density=self.poleMass,
            ),
        )
        
        # Initial hinge joint between cart and pole
        self.world.CreateRevoluteJoint(
            bodyA=self.cart,
            bodyB=self.pole,
            anchor=self.cart.worldCenter,
            enableMotor=False,
        )

    def get_state(self):
        """Return the current state of the cart-pole system."""
        position = self.cart.position.x
        velocity = self.cart.linearVelocity.x
        angle = self.pole.angle
        angular_velocity = self.pole.angularVelocity

        return [position, velocity, angle, angular_velocity]

    def act(self, action):
        """Apply a force to the cart based on the action value: -1 (left), 0 (no movement), or 1 (right)."""
        force = self.force * action  # Multiply force by action (-1, 0, or 1)
        self.cart.ApplyForceToCenter((force, 0), wake=True)
        
        # Step the world for physics simulation
        timeStep = 1.0 / self.framesPerSecond
        self.world.Step(timeStep, self.velocityIterations, self.positionIterations)

    def render(self):
        """Render the cart-pole system (optional for visual debugging)."""
        # Initialize Pygame screen and other settings if needed for visualization
        pygame.init()
        screen = pygame.display.set_mode(self.screenSize)
        clock = pygame.time.Clock()

        # Render loop (optional, mainly for debugging purposes)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
            
            screen.fill((255, 255, 255))  # Clear screen with white

            # Add rendering logic here if needed
            # Draw track, cart, and pole

            pygame.display.flip()
            clock.tick(self.framesPerSecond)
            
if __name__ == "__main__":
    cart_pole = CartPole()  # Initialize the CartPole environment
    cart_pole.render()       # Call the render function to visualize the environment


    pygame.quit()
