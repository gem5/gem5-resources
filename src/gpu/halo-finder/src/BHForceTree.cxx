/*=========================================================================
                                                                                
Copyright (c) 2007, Los Alamos National Security, LLC

All rights reserved.

Copyright 2007. Los Alamos National Security, LLC. 
This software was produced under U.S. Government contract DE-AC52-06NA25396 
for Los Alamos National Laboratory (LANL), which is operated by 
Los Alamos National Security, LLC for the U.S. Department of Energy. 
The U.S. Government has rights to use, reproduce, and distribute this software. 
NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY,
EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  
If software is modified to produce derivative works, such modified software 
should be clearly marked, so as not to confuse it with the version available 
from LANL.
 
Additionally, redistribution and use in source and binary forms, with or 
without modification, are permitted provided that the following conditions 
are met:
-   Redistributions of source code must retain the above copyright notice, 
    this list of conditions and the following disclaimer. 
-   Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution. 
-   Neither the name of Los Alamos National Security, LLC, Los Alamos National
    Laboratory, LANL, the U.S. Government, nor the names of its contributors
    may be used to endorse or promote products derived from this software 
    without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
                                                                                
=========================================================================*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>
#include <math.h>

#include <time.h>

#include "Timings.h"
#include "BHForceTree.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////
//
// FParticle contains information about particles
//
/////////////////////////////////////////////////////////////////////////

FParticle::FParticle()
{
  this->parent = -1;
  this->nextNode = -1;
  this->sibling = -1;
  this->force = 0.0;
}

/////////////////////////////////////////////////////////////////////////
//
// FNode is a region of physical space divided into octants
//
/////////////////////////////////////////////////////////////////////////

FNode::FNode(POSVEL_T* minLoc, POSVEL_T* maxLoc)
{
  for (int dim = 0; dim < DIMENSION; dim++) {
    this->geoSide[dim] = maxLoc[dim] - minLoc[dim];
    this->geoCenter[dim] = minLoc[dim] + this->geoSide[dim] * 0.5;

  }
  for (int i = 0; i < NUM_CHILDREN; i++)
    this->u.child[i] = -1;
}

/////////////////////////////////////////////////////////////////////////
//
// FNode constructed from an octant of a parent node
//
/////////////////////////////////////////////////////////////////////////

FNode::FNode(FNode* parent, int oindx)
{
  for (int dim = 0; dim < DIMENSION; dim++) {
    this->geoSide[dim] = parent->geoSide[dim] * 0.5;
  }

  // Vary Z fastest when making octtree children
  // If this changes must also change getChildIndex()
  if (oindx & 4)
    this->geoCenter[0] = parent->geoCenter[0] + this->geoSide[0] * 0.5;
  else
    this->geoCenter[0] = parent->geoCenter[0] - this->geoSide[0] * 0.5;

  if (oindx & 2)
    this->geoCenter[1] = parent->geoCenter[1] + this->geoSide[1] * 0.5;
  else
    this->geoCenter[1] = parent->geoCenter[1] - this->geoSide[1] * 0.5;

  if (oindx & 1)
    this->geoCenter[2] = parent->geoCenter[2] + this->geoSide[2] * 0.5;
  else
    this->geoCenter[2] = parent->geoCenter[2] - this->geoSide[2] * 0.5;

  for (int i = 0; i < NUM_CHILDREN; i++)
    this->u.child[i] = -1;
}

/////////////////////////////////////////////////////////////////////////
//
// Barnes Hut Tree
//
/////////////////////////////////////////////////////////////////////////

BHForceTree::BHForceTree(
			 POSVEL_T* minLoc,
			 POSVEL_T* maxLoc,
			 ID_T count,
			 POSVEL_T* xLoc,
			 POSVEL_T* yLoc,
			 POSVEL_T* zLoc,
			 POSVEL_T* xVel,
			 POSVEL_T* yVel,
			 POSVEL_T* zVel,
			 POSVEL_T* ms,
			 POSVEL_T avgMass)
{
  // Extract the contiguous data block from a vector pointer
  this->particleCount = count;
  this->nodeOffset = this->particleCount;
  this->xx = xLoc;
  this->yy = yLoc;
  this->zz = zLoc;
  this->vx = xVel;
  this->vy = yVel;
  this->vz = zVel;
  this->mass = ms;
  this->particleMass = avgMass;

  // Find the grid size of this chaining mesh
  for (int dim = 0; dim < DIMENSION; dim++) {
    this->minRange[dim] = minLoc[dim];
    this->maxRange[dim] = maxLoc[dim];
  }
  this->boxSize = this->maxRange[0] - this->minRange[0];

  //maybe change this to Newton's law or something
  this->m_fl = new ForceLawNewton();
  this->m_fcoeff = 1.0;

  // Create the recursive BH tree from the particle locations
  createBHForceTree();

  // Thread the recursive tree turning it into an iterative tree
  ID_T rootIndx = this->particleCount;
  ID_T sibling = -1;
  ID_T parent = -1;
  ID_T lastIndx = -1;
  POSVEL_T radius = 0.0;

  threadBHForceTree(rootIndx, sibling, parent, &lastIndx, &radius);
}

BHForceTree::BHForceTree(
			 POSVEL_T* minLoc,
			 POSVEL_T* maxLoc,
			 ID_T count,
			 POSVEL_T* xLoc,
			 POSVEL_T* yLoc,
			 POSVEL_T* zLoc,
			 POSVEL_T* xVel,
			 POSVEL_T* yVel,
			 POSVEL_T* zVel,
			 POSVEL_T* ms,
			 POSVEL_T avgMass,
			 ForceLaw *fl,
			 float fcoeff)
{
  // Extract the contiguous data block from a vector pointer
  this->particleCount = count;
  this->nodeOffset = this->particleCount;
  this->xx = xLoc;
  this->yy = yLoc;
  this->zz = zLoc;
  this->vx = xVel;
  this->vy = yVel;
  this->vz = zVel;
  this->mass = ms;
  this->particleMass = avgMass;

  // Find the grid size of this chaining mesh
  for (int dim = 0; dim < DIMENSION; dim++) {
    this->minRange[dim] = minLoc[dim];
    this->maxRange[dim] = maxLoc[dim];
  }
  this->boxSize = this->maxRange[0] - this->minRange[0];

  this->m_fl = fl;
  this->m_fcoeff = fcoeff;

  // Create the recursive BH tree from the particle locations
  createBHForceTree();

  // Thread the recursive tree turning it into an iterative tree
  ID_T rootIndx = this->particleCount;
  ID_T sibling = -1;
  ID_T parent = -1;
  ID_T lastIndx = -1;
  POSVEL_T radius = 0.0;
  threadBHForceTree(rootIndx, sibling, parent, &lastIndx, &radius);
}

BHForceTree::~BHForceTree()
{
  /* empty */
}

/////////////////////////////////////////////////////////////////////////
//
// Find the subhalos of the FOF halo using SUBFIND algorithm which
// requires subhalos to be locally overdense and self-bound
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::createBHForceTree()
{
  // Create the FParticles
  this->fParticle.resize(this->particleCount);

  // Reserve a basic amount of space in the BH tree
  this->fNode.reserve(this->particleCount/NUM_CHILDREN);

  // Create the root node of the BH tree
  FNode root(this->minRange, this->maxRange);
  this->fNode.push_back(root);
  ID_T nodeIndex = 0;

  // Iterate on all particles placing them in the BH tree
  // Child slots in the tree contain the index of the FParticle or
  // the index of the FNode offset by the number of particles
  // This is so we can use an integer instead of pointers to refer to objects
  //
  for (ID_T pindx = 0; pindx < this->particleCount; pindx++) {

    // Start at root of tree for insertion of a new particle
    // pindx is index into the halo particles where location is stored
    // tindx is index into the BH tree nodes
    // oindx is index into the octant of the tree node
    ID_T tindx = 0;
    int oindx = getChildIndex(&this->fNode[tindx], pindx);

    while (this->fNode[tindx].u.child[oindx] != -1) {

      // Child slot in tree contains another SPHNode so go there
      if (this->fNode[tindx].u.child[oindx] > this->particleCount) {
        tindx = this->fNode[tindx].u.child[oindx] - this->particleCount;
        oindx = getChildIndex(&this->fNode[tindx], pindx);
      }

      // Otherwise there is a particle in the slot and we make a new FNode
      else {

        // Get the particle index of particle already in the node
        ID_T pindx2 = this->fNode[tindx].u.child[oindx];

	// First, check to make sure that this particle is not at the exact
	// same location as the particle that is already there. If it is, then
	// we'll double the mass of the existing particle and leave this one
	// out.
        if (this->xx[pindx2] == this->xx[pindx] &&
            this->yy[pindx2] == this->yy[pindx] &&
            this->zz[pindx2] == this->zz[pindx]) {
          this->mass[pindx2] += this->mass[pindx];
          goto next_particle;
        }

 
        // Make sure that the vector does not over allocate
        if (this->fNode.capacity() == this->fNode.size()) {
          this->fNode.reserve(this->fNode.capacity() 
            + this->particleCount/NUM_CHILDREN);
        }

        FNode node(&this->fNode[tindx], oindx);
        this->fNode.push_back(node);
        nodeIndex++;
        ID_T tindx2 = nodeIndex;
        
        // Place the node that was sitting there already
        int oindx2 = getChildIndex(&this->fNode[tindx2], pindx2);
        this->fNode[tindx2].u.child[oindx2] = pindx2;

        // Add the new SPHNode to the BHTree
        this->fNode[tindx].u.child[oindx] = tindx2 + this->particleCount;

        // Set to new node
        tindx = tindx2;
        oindx = getChildIndex(&this->fNode[tindx], pindx);
      }
    }
    // Place the current particle in the BH tree
    this->fNode[tindx].u.child[oindx] = pindx;
next_particle:;
  }
  this->nodeCount = this->fNode.size();
}

/////////////////////////////////////////////////////////////////////////
//
// Update the FNode vector by walking using a depth first recursion
// Set parent and sibling indices which can replace the child[8] already
// there, and supply extra information about center of mass, total particle
// mass and particle radius which is the distance from the center of mass
// to the furthest particle.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::threadBHForceTree(
			ID_T curIndx,	     // Current node/particle
			ID_T sibling,        // Sibling of current
			ID_T parent,         // Parent of current
			ID_T* lastIndx,      // Last node/particle
                        POSVEL_T* radius)    // Needed to pass up partRadius
{
  // Set the next index in the threading for node or particle
  // Particles and nodes are threaded together so all are touched in iteration
  if (*lastIndx >= 0) {
    if (*lastIndx >= this->nodeOffset)
      this->fNode[(*lastIndx - this->nodeOffset)].u.n.nextNode = curIndx;
    else
      this->fParticle[*lastIndx].nextNode = curIndx;
  }
  *lastIndx = curIndx;
 
  // FParticle saves the parent and sibling FNode id
  if (curIndx < this->nodeOffset) {
    this->fParticle[curIndx].parent = parent;
    this->fParticle[curIndx].sibling = sibling;

  // FNode recurses on each of the children
  } else {
    ID_T child[NUM_CHILDREN];
    FNode* curNode = &this->fNode[curIndx - this->nodeOffset];

    // Store mass and center of mass for each child node or particle
    POSVEL_T childMass[NUM_CHILDREN];
    POSVEL_T childRadius[NUM_CHILDREN];
    POSVEL_T childCenter[NUM_CHILDREN][DIMENSION];
    for (int j = 0; j < NUM_CHILDREN; j++) {
      child[j] = curNode->u.child[j];
      childMass[j] = 0.0;
      childRadius[j] = 0.0;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Recurse on each of the children, recording information on the way up
    //
    for (int j = 0; j < NUM_CHILDREN; j++) {

      // Skip any children which contain neither a particle or node
      ID_T childIndx, childIndxNext, nextSibling;
      if ((childIndx = child[j]) >= 0) {

        // Check for a sibling on this level or set to the next level up
        int jj;
        for (jj = j + 1; jj < NUM_CHILDREN; jj++)
          if ((childIndxNext = child[jj]) >= 0)
            break;
        if (jj < NUM_CHILDREN)
          nextSibling = childIndxNext;
        else
          nextSibling = -1;

        // Recursion to child
        // Since value of partRadius set in child is not necessarily the
        // distance between center of mass and futhest child return it
        threadBHForceTree(childIndx, nextSibling, curIndx, lastIndx, radius);

        // Child is a node or a particle
        if (childIndx >= this->nodeOffset) {

          // FNode, gather mass and center of mass of all contained particles
          FNode* childNode = &this->fNode[childIndx - this->nodeOffset];
          childMass[j] = childNode->u.n.partMass;
          childRadius[j] = *radius;
          for (int dim = 0; dim < DIMENSION; dim++)
            childCenter[j][dim] = childNode->u.n.partCenter[dim];

        } else {
          // FParticle, set mass and center of mass using particle location
          childMass[j] = this->particleMass;
          childRadius[j] = 0.0;
          childCenter[j][0] = this->xx[childIndx];
          childCenter[j][1] = this->yy[childIndx];
          childCenter[j][2] = this->zz[childIndx];
        }
      }
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Finished processing all children, collect information for this node
    //
    curNode->u.n.partMass = 0.0;
    for (int dim = 0; dim < DIMENSION; dim++)
      curNode->u.n.partCenter[dim] = 0.0;

    // Collect total mass and total center of mass for all children
    for (int j = 0; j < NUM_CHILDREN; j++) {
      if (childMass[j] > 0) {
        curNode->u.n.partMass += childMass[j];
        for (int dim = 0; dim < DIMENSION; dim++)
          curNode->u.n.partCenter[dim] += 
                                     childCenter[j][dim] * childMass[j];
      }
    }

    // Calculate center of mass for current node
    if (curNode->u.n.partMass > 0.0) {
      for (int dim = 0; dim < DIMENSION; dim++)
        curNode->u.n.partCenter[dim] /= curNode->u.n.partMass;
    } else {
      for (int dim = 0; dim < DIMENSION; dim++)
        curNode->u.n.partCenter[dim] = curNode->geoCenter[dim];
    }

    // First method for calculating particle radius
    // Calculate the radius from node center of mass to furthest node corner
    POSVEL_T partRadius1 = distanceToFarCorner(
                                     curNode->u.n.partCenter[0],
                                     curNode->u.n.partCenter[1],
                                     curNode->u.n.partCenter[2],
                                     curNode);

    // Second method for calculating particle radius
    // Calculate the radius from center of mass to furthest child
    POSVEL_T partRadius2 = 0.0;
    for (int j = 0; j < NUM_CHILDREN; j++) {
      if (childMass[j] > 0.0) {

        // Calculate the distance between this center of mass and that of child
        POSVEL_T dist = distanceToCenterOfMass(childCenter[j][0],
                                               childCenter[j][1],
                                               childCenter[j][2],
                                               curNode);
        // Add in the particle radius of the child to get furthest point
        dist += childRadius[j];
        if (dist > partRadius2)
          partRadius2 = dist;
      }
    }
    // Used by parent of this node
    *radius = partRadius2;

    // Save the smaller of the two particle radii
    curNode->u.n.partRadius = min(partRadius1, partRadius2);

    // Set threaded structure for this node
    curNode->u.n.sibling = sibling;
    curNode->u.n.parent = parent;
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Particle i has location vector x_i (xx_i, yy_i, zz_i)
// Particle i has velocity vector v_i (vx_i, vy_i, vz_i)
//
// Node j has center of particle mass c_j (cx_j, cy_j, cz_j)
// Node j has total particle mass of M_j
// Node j has bounding radius of R_j
//
// Distance between particle i and node j is
//   d_ji = fabs(c_j - x_i)
//
// Rule for updating
//   v_i(t_1) = v_i(t_0) + alpha * Sum over j of f_ji(d_ji, Mj)
//          where f_ji is the short range force over finite range r_f
//          where alpha is some coeffient
//          where Sum over j nodes is determined by a tree walk
//
// An opening angle is defined as
//    theta_ji = (2 R_j) / d_ji
//
// This angle determines whether a node should be opened to a higher resolution
// or whether it can be used as is because it is small enough or far enough away
// This is determined by comparing to a passed in theta_0
//
// Three actions can occur for a node encountered on the walk
//
//   1. Node is too far away to contribute to force
//      if d_ji - R_j > r_f
//      or distance of x_i to nearest cornder of node > r_f
//
//   2. Node is close enough to contribute so check the opening angle
//      if theta_ji > theta_0 follow nextNode to open this node to children
//
//   3. Node is close enough and theta_ji < theta_0
//      calculate f_ji(d_ji, Mj) and update v_i
//      follow the sibling link and not the nextNode link
//
// Force is calculated for each particle i by
//   Starting at the root node and walking the entire tree collecting force
//   Starting at the particle and walking up parents until a criteria is met
//
/////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////
//
// Short range force full N^2 calculation
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::treeForceN2(
                    POSVEL_T critRadius)  // Radius of furthest point
{
  POSVEL_T critRadius2 = critRadius * critRadius;

  POTENTIAL_T* force = new POTENTIAL_T[this->particleCount];
  for (int i = 0; i < this->particleCount; i++)
    force[i] = 0.0;

  // First particle in halo to calculate force on
  for (int p = 0; p < this->particleCount; p++) {

    // Next particle in halo force loop
    for (int q = p+1; q < this->particleCount; q++) {

      POSVEL_T dx = (POSVEL_T) fabs(this->xx[p] - this->xx[q]);
      POSVEL_T dy = (POSVEL_T) fabs(this->yy[p] - this->yy[q]);
      POSVEL_T dz = (POSVEL_T) fabs(this->zz[p] - this->zz[q]);
      POSVEL_T r2 = dx * dx + dy * dy + dz * dz;

      if (r2 != 0.0 && r2 < critRadius2) {
        force[p] -= (this->mass[q] / r2);
        force[q] -= (this->mass[p] / r2);
      }
    }
  }
  for (int p = 0; p < this->particleCount; p++) {
    this->fParticle[p].force = force[p];
  }
  delete [] force;
}

/////////////////////////////////////////////////////////////////////////
//
// Short range gravity calculation for group of particles in a node
// Walk down the tree from the root until reaching node with less than the
// maximum number of particles in a group.  Create an interaction list that
// will work for all particles and calculate force.  For particles within
// the group the calculation will be n^2.  For nodes outside the group
// decisions are made on whether to include the node or ignore it, or
// to accept it or open it.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::treeForceGroup(
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius,  // Accept or ignore node not opened
                    int minGroup,         // Minimum particles in a group
                    int maxGroup)         // Maximum particles in a group
{
  ID_T root = this->particleCount;
  POSVEL_T maxMass = maxGroup * this->particleMass;
  POSVEL_T minMass = minGroup * this->particleMass;

  walkTreeGroup(root, minMass, maxMass, bhAngle, critRadius);
}

/////////////////////////////////////////////////////////////////////////
//
// Walk the tree in search of nodes which are less than the maximum
// number of particles to constitute a group.  All particles in the group
// will be treated together with the n^2 force calculated between members
// of the group and then having an interaction list applied.  The group
// may consist of other nodes and particles and so the recursive descent
// will be needed to find all particles in the group.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::walkTreeGroup(
                    ID_T curId,           // Particle to calculate force on
                    POSVEL_T minMass,     // Minimum mass for a group
                    POSVEL_T maxMass,     // Maximum mass for a group
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius)  // Accept or ignore node not opened
{

  if (curId < this->nodeOffset) {
    // Current object is a particle
    vector<ID_T>* partInteract = new vector<ID_T>;
    vector<ID_T>* nodeInteract = new vector<ID_T>;

    createParticleInteractList(curId, bhAngle, critRadius,
                               partInteract, nodeInteract);
    this->fParticle[curId].force = forceCalculation(
                                      curId, partInteract, nodeInteract);
    delete partInteract;
    delete nodeInteract;
  }
  else {
    // Current object is a node
    ID_T child = this->fNode[curId - this->nodeOffset].u.n.nextNode;
    while (child != -1) {
      if (child < this->nodeOffset) {
        // Child is a particle
        vector<ID_T>* partInteract = new vector<ID_T>;
        vector<ID_T>* nodeInteract = new vector<ID_T>;

        createParticleInteractList(child, bhAngle, critRadius,
                                   partInteract, nodeInteract);
        this->fParticle[child].force = forceCalculation(
                                          child, partInteract, nodeInteract);
        child = this->fParticle[child].sibling;

        delete partInteract;
        delete nodeInteract;
      }
      else {
        // Child is a node
        FNode* childNode = &this->fNode[child - this->nodeOffset];
        if (childNode->u.n.partMass < maxMass &&
            childNode->u.n.partRadius < (critRadius * 0.5)) {

          // If the group is too small it can't function as a group
          // so run the topdown method on those particles
          if (childNode->u.n.partMass < minMass) {

            // Collect particles in subgroup
            vector<ID_T>* particles = new vector<ID_T>;
            collectParticles(child, particles);
            int count = particles->size();

            for (int i = 0; i < count; i++) {
              ID_T pId = (*particles)[i];
              treeForceGadgetTopDown(pId, bhAngle, critRadius);
            }
          }
          else {

            vector<ID_T>* partInteract = new vector<ID_T>;
            vector<ID_T>* nodeInteract = new vector<ID_T>;

            createNodeInteractList(child, bhAngle, critRadius,
                                   partInteract, nodeInteract);
            forceCalculationGroup(child, bhAngle, critRadius,
                                  partInteract, nodeInteract);
            delete partInteract;
            delete nodeInteract;
          }
        }
        else {
          walkTreeGroup(child, minMass, maxMass, bhAngle, critRadius);
        }
        child = this->fNode[child - this->nodeOffset].u.n.sibling;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Create the interaction list for the particle starting at root
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::createParticleInteractList(
                    ID_T p,               // Particle to calculate force on
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius , // Accept or ignore node not opened
                    vector<ID_T>* partInteract,
                    vector<ID_T>* nodeInteract)
{
  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;
  POSVEL_T pos_x = this->xx[p];
  POSVEL_T pos_y = this->yy[p];
  POSVEL_T pos_z = this->zz[p];

  // Follow thread through tree from root choosing nodes and particles
  // which will contribute to the force of the given particle
  ID_T root = this->particleCount;
  ID_T index = root;

  while (index >= 0) {

    if (index < this->nodeOffset) {
      // Particle
      dx = this->xx[index] - pos_x;
      dy = this->yy[index] - pos_y;
      dz = this->zz[index] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      if (r < critRadius) {
        partInteract->push_back(index);
      }
      index = this->fParticle[index].nextNode;
    }

    else {
      // Node
      FNode* curNode = &this->fNode[index - this->nodeOffset];
      partRadius = curNode->u.n.partRadius;
      distToNearPoint = distanceToNearestPoint(pos_x, pos_y, pos_z, curNode);

      dx = curNode->u.n.partCenter[0] - pos_x;
      dy = curNode->u.n.partCenter[1] - pos_y;
      dz = curNode->u.n.partCenter[2] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      // Node is ignored if it is too far away from the particle
      // Distance from particle to particle radius exceeds critical radius
      // Distance from particle to nearest side of node exceeds critical radius

      if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {

        // Ignore node, move on to sibling of this node
        index = curNode->u.n.sibling;

        // If there is no sibling go up a level until we find a node
        ID_T parent = curNode->u.n.parent;
        while (index == -1 && parent != -1 && parent != root) {
          index = this->fNode[parent - this->nodeOffset].u.n.sibling;
          parent = this->fNode[parent - this->nodeOffset].u.n.parent;
        }
      }
      else {
        if (2*partRadius > (r * bhAngle)) {
          // Open node
          index = curNode->u.n.nextNode;
        } else {
          // Accept
          nodeInteract->push_back(index);
          index = curNode->u.n.sibling;

          // If there is no sibling go up a level until we find a node
          ID_T parent = curNode->u.n.parent;
          while (index == -1 && parent != -1 && parent != root) {
            index = this->fNode[parent - this->nodeOffset].u.n.sibling;
            parent = this->fNode[parent - this->nodeOffset].u.n.parent;
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Create the interaction list for the node starting at root
// must test for acceptance based on a radius from center of mass to
// furthest particle to make sure it is most inclusive.
// Make sure my definition of partRadius does this
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::createNodeInteractList(
                    ID_T node,
                    POSVEL_T bhAngle,    // Open node to examine children
                    POSVEL_T critRadius, // Accept or ignore node not opened
                    vector<ID_T>* partInteract,
                    vector<ID_T>* nodeInteract)
{
  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;
  FNode* curNode = &this->fNode[node - this->nodeOffset];
  POSVEL_T pos_x = curNode->u.n.partCenter[0];
  POSVEL_T pos_y = curNode->u.n.partCenter[1];
  POSVEL_T pos_z = curNode->u.n.partCenter[2];

  // Follow thread through tree from root choosing nodes and particles
  // which will contribute to the force of the given particle
  ID_T root = this->particleCount;
  ID_T index = root;

  while (index >= 0) {

    if (index < this->nodeOffset) {
      // Particle
      dx = this->xx[index] - pos_x;
      dy = this->yy[index] - pos_y;
      dz = this->zz[index] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      if (r < critRadius) {
        partInteract->push_back(index);
      }
      index = this->fParticle[index].nextNode;
    }

    else {
      // Node
      FNode* childNode = &this->fNode[index - this->nodeOffset];

      // If the child is the node we are building the list for skip
      if (childNode != curNode) {
        partRadius = childNode->u.n.partRadius;
        distToNearPoint = distanceToNearestPoint(
                            pos_x, pos_y, pos_z, childNode);

        dx = childNode->u.n.partCenter[0] - pos_x;
        dy = childNode->u.n.partCenter[1] - pos_y;
        dz = childNode->u.n.partCenter[2] - pos_z;
        r = sqrt(dx * dx + dy * dy + dz * dz);

        if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {

          // Ignore node, move on to sibling of this node
          index = childNode->u.n.sibling;

          // If there is no sibling go up a level until we find a node
          ID_T parent = childNode->u.n.parent;
          while (index == -1 && parent != -1 && parent != root) {
            index = this->fNode[parent - this->nodeOffset].u.n.sibling;
            parent = this->fNode[parent - this->nodeOffset].u.n.parent;
          }
        }
        else {
          if (2*partRadius > (r * bhAngle)) {
            // Open node
            index = childNode->u.n.nextNode;
          } else {
            // Accept
            nodeInteract->push_back(index);
            index = childNode->u.n.sibling;

            // If there is no sibling go up a level until we find a node
            ID_T parent = childNode->u.n.parent;
            while (index == -1 && parent != -1 && parent != root) {
              index = this->fNode[parent - this->nodeOffset].u.n.sibling;
              parent = this->fNode[parent - this->nodeOffset].u.n.parent;
            }
          }
        }
      }
      else {
        index = childNode->u.n.sibling;
        ID_T parent = childNode->u.n.parent;
        while (index == -1 && parent != -1 && parent != root) {
          index = this->fNode[parent-nodeOffset].u.n.sibling;
          parent = this->fNode[parent-nodeOffset].u.n.parent;
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Force calculation for a group of particles
// Force is calculated between every pair of particles in the group
// Interaction lists are applied to every particle in the group
// Tree walk will hav to continue from this node to locate all the
// particles which might be in subnodes of this node.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::forceCalculationGroup(
                    ID_T node,
                    POSVEL_T bhAngle,    // Open or accepc
                    POSVEL_T critRadius, // Accept or ignore node not opened
                    vector<ID_T>* partInteract,
                    vector<ID_T>* nodeInteract)
{
  // Collect all particles in the tree from this node downwards
  vector<ID_T>* particles = new vector<ID_T>;
  collectParticles(node, particles);
  int count = particles->size();

  // Process each particle against all others in the group
  // Use the minimumPotential() code to make this n^2/2 for upper triangular
  // Arrange in an upper triangular grid to save computation
  POTENTIAL_T* force = new POTENTIAL_T[count];
  for (int i = 0; i < count; i++)
    force[i] = 0.0;

  // First particle in halo to calculate force on
  for (int p = 0; p < count; p++) {

    // Next particle in halo force loop
    for (int q = p+1; q < count; q++) {

      ID_T pId = (*particles)[p];
      ID_T qId = (*particles)[q];

      POSVEL_T dx = (POSVEL_T) fabs(this->xx[pId] - this->xx[qId]);
      POSVEL_T dy = (POSVEL_T) fabs(this->yy[pId] - this->yy[qId]);
      POSVEL_T dz = (POSVEL_T) fabs(this->zz[pId] - this->zz[qId]);
      POSVEL_T r2 = dx * dx + dy * dy + dz * dz;

      if (r2 != 0.0) {
        force[p] -= (this->mass[qId] / r2);
        force[q] -= (this->mass[pId] / r2);
      }
    }
  }

  // Process each particle against the interaction lists
  // Node interact list was created using the node this particle is in
  // so it may need to be adjusted first
  for (int p = 0; p < count; p++) {
    ID_T pId = (*particles)[p];

    POSVEL_T value = 
      forceCalculationParticle(pId, critRadius, 
                               partInteract, nodeInteract);
    force[p] += value;
    this->fParticle[pId].force = force[p];
  }
  delete particles;
  delete [] force;
}

/////////////////////////////////////////////////////////////////////////
//
// Short range force calculation
// Potential is calculated and is used to determine the acceleration of
// the particle.  Acceleration is applied to the current velocity to
// produce the velocity at the next time step.
//
/////////////////////////////////////////////////////////////////////////

POSVEL_T BHForceTree::forceCalculationParticle(
                    ID_T p0,                    // Target particle index
                    POSVEL_T critRadius,
                    vector<ID_T>* partInteract, // Particles acting on p
                    vector<ID_T>* nodeInteract) // Nodes acting on p
{
  POSVEL_T accel[DIMENSION];
  POSVEL_T phi = 0.0;

  POSVEL_T pos0_x = this->xx[p0];
  POSVEL_T pos0_y = this->yy[p0];
  POSVEL_T pos0_z = this->zz[p0];

  for (int dim = 0; dim < DIMENSION; dim++)
    accel[dim] = 0.0;

  int numberOfNodes = (int) nodeInteract->size();
  int numberOfParticles = (int) partInteract->size();

  // Particles contributing to the force use location and mass of one particle
  for (int p = 0; p < numberOfParticles; p++) {
    ID_T particle = (*partInteract)[p];
    if (p0 != particle) {
      POSVEL_T dx = this->xx[particle] - pos0_x;
      POSVEL_T dy = this->yy[particle] - pos0_y;
      POSVEL_T dz = this->zz[particle] - pos0_z;

      POSVEL_T r2 = dx * dx + dy * dy + dz * dz;
      POSVEL_T r = sqrt(r2);

      if (r < critRadius) {
        POSVEL_T f_over_r = this->mass[particle] * m_fl->f_over_r(r2);
        //POSVEL_T f_over_r = this->mass[particle] / r2;
        phi -= f_over_r;

        accel[0] += dx * f_over_r * m_fcoeff;
        accel[1] += dy * f_over_r * m_fcoeff;
        accel[2] += dz * f_over_r * m_fcoeff;

        this->vx[p0] += dx * f_over_r * m_fcoeff;
        this->vy[p0] += dy * f_over_r * m_fcoeff;
        this->vz[p0] += dz * f_over_r * m_fcoeff;
      }
    }
  }

  // Nodes contributing to force use center of mass and total particle mass
  for (int n = 0; n < numberOfNodes; n++) {
    FNode* node = &this->fNode[(*nodeInteract)[n] - this->nodeOffset];
    POSVEL_T partRadius = node->u.n.partRadius;
    POSVEL_T distToNearPoint = distanceToNearestPoint(
                                 pos0_x, pos0_y, pos0_z, node);

    POSVEL_T dx = node->u.n.partCenter[0] - pos0_x;
    POSVEL_T dy = node->u.n.partCenter[1] - pos0_y;
    POSVEL_T dz = node->u.n.partCenter[2] - pos0_z;

    POSVEL_T r2 = dx * dx + dy * dy + dz * dz;
    POSVEL_T r = sqrt(r2);

    if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
      // Ignore
    } else {
      POSVEL_T f_over_r = node->u.n.partMass * m_fl->f_over_r(r2);
      //POSVEL_T f_over_r = node->u.n.partMass / r2;
      phi -= f_over_r;

      accel[0] += dx * f_over_r * m_fcoeff;
      accel[1] += dy * f_over_r * m_fcoeff;
      accel[2] += dz * f_over_r * m_fcoeff;
    
      this->vx[p0] += dx * f_over_r * m_fcoeff;
      this->vy[p0] += dy * f_over_r * m_fcoeff;
      this->vz[p0] += dz * f_over_r * m_fcoeff;
    }
  }
  return phi;
}


/////////////////////////////////////////////////////////////////////////
//
// Collect all particle ids from this node downwards in tree
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::collectParticles(ID_T curId, vector<ID_T>* particles)
{
  FNode* curNode = &this->fNode[curId - this->nodeOffset];
  ID_T child = curNode->u.n.nextNode;
  while (child != -1) {
    if (child < this->nodeOffset) {
      particles->push_back(child);
      child = this->fParticle[child].sibling;
    } else {
      collectParticles(child, particles);
      child = this->fNode[child - this->nodeOffset].u.n.sibling;
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Short range gravity calculation for a single particle
// Starting with the root and following threads and siblings makes decisions
// about which nodes are opened, accepted or ignored
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::treeForceGadgetTopDown(
                    ID_T p,               // Particle to calculate force on
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius)  // Accept or ignore node not opened
{
  // Keep vectors for now but eventually force can be accumulated
  // on all particles and on nodes that have been accepted
  vector<ID_T>* partInteract = new vector<ID_T>;
  vector<ID_T>* nodeInteract = new vector<ID_T>;

  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;
  POSVEL_T pos_x = this->xx[p];
  POSVEL_T pos_y = this->yy[p];
  POSVEL_T pos_z = this->zz[p];
  
  // Follow thread through tree from root choosing nodes and particles
  // which will contribute to the force of the given particle
  ID_T root = this->particleCount;
  ID_T index = root;

  while (index >= 0) {

    if (index < this->nodeOffset) {
      // Particle
      dx = this->xx[index] - pos_x;
      dy = this->yy[index] - pos_y;
      dz = this->zz[index] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      if (r < critRadius) {
        partInteract->push_back(index);
      }
      index = this->fParticle[index].nextNode;
    } else {
      // Node
      FNode* curNode = &this->fNode[index - this->nodeOffset];
      partRadius = curNode->u.n.partRadius;
      distToNearPoint = distanceToNearestPoint(pos_x, pos_y, pos_z, curNode);
      
      dx = curNode->u.n.partCenter[0] - pos_x;
      dy = curNode->u.n.partCenter[1] - pos_y;
      dz = curNode->u.n.partCenter[2] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);
      
      // Node is ignored if it is too far away from the particle
      // Distance from particle to particle radius exceeds critical radius
      // Distance from particle to nearest side of node exceeds critical radius
      if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
	
        // Ignore node, move on to sibling of this node
        index = curNode->u.n.sibling;
	
        // If there is no sibling go up a level until we find a node
        ID_T parent = curNode->u.n.parent;
        while (index == -1 && parent != -1 && parent != root) {
          index = this->fNode[parent - this->nodeOffset].u.n.sibling;
          parent = this->fNode[parent - this->nodeOffset].u.n.parent;
        }
      } else { 
        if (2*partRadius > (r * bhAngle)) { 
          // Open node, move on to first child
          index = curNode->u.n.nextNode;

        } else {
          // Accept node, add to interact list, move on to sibling
          nodeInteract->push_back(index);
          index = curNode->u.n.sibling;
 
          // If there is no sibling go up a level until we find a node
          ID_T parent = curNode->u.n.parent;
          while (index == -1 && parent != -1 && parent != root) {
            index = this->fNode[parent-nodeOffset].u.n.sibling;
            parent = this->fNode[parent-nodeOffset].u.n.parent;
          }
        }
      }
    }
  }

  // Force calculation for this particle
  this->fParticle[p].force = 
    forceCalculation(p, partInteract, nodeInteract);

  delete partInteract;
  delete nodeInteract;
}


void BHForceTree::treeForceGadgetTopDownFast(
                    ID_T p,               // Particle to calculate force on
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius)  // Accept or ignore node not opened
{
  vector<POSVEL_T>* xInteract = new vector<POSVEL_T>;
  vector<POSVEL_T>* yInteract = new vector<POSVEL_T>;
  vector<POSVEL_T>* zInteract = new vector<POSVEL_T>;
  vector<POSVEL_T>* mInteract = new vector<POSVEL_T>;

  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;
  POSVEL_T pos_x = this->xx[p];
  POSVEL_T pos_y = this->yy[p];
  POSVEL_T pos_z = this->zz[p];
  
  // Follow thread through tree from root choosing nodes and particles
  // which will contribute to the force of the given particle
  ID_T root = this->particleCount;
  ID_T index = root;

  while (index >= 0) {

    if (index < this->nodeOffset) {
      // Particle
      dx = this->xx[index] - pos_x;
      dy = this->yy[index] - pos_y;
      dz = this->zz[index] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      if (r < critRadius && p != index) {
	xInteract->push_back(this->xx[index]);
	yInteract->push_back(this->yy[index]);
	zInteract->push_back(this->zz[index]);
	mInteract->push_back(this->mass[index]);
      }

      index = this->fParticle[index].nextNode;
    } else {
      // Node
      FNode* curNode = &this->fNode[index - this->nodeOffset];
      partRadius = curNode->u.n.partRadius;
      distToNearPoint = distanceToNearestPoint(pos_x, pos_y, pos_z, curNode);
      
      dx = curNode->u.n.partCenter[0] - pos_x;
      dy = curNode->u.n.partCenter[1] - pos_y;
      dz = curNode->u.n.partCenter[2] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);
      
      // Node is ignored if it is too far away from the particle
      // Distance from particle to particle radius exceeds critical radius
      // Distance from particle to nearest side of node exceeds critical radius
      if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
	
        // Ignore node, move on to sibling of this node
        index = curNode->u.n.sibling;
	
        // If there is no sibling go up a level until we find a node
        ID_T parent = curNode->u.n.parent;
        while (index == -1 && parent != -1 && parent != root) {
          index = this->fNode[parent - this->nodeOffset].u.n.sibling;
          parent = this->fNode[parent - this->nodeOffset].u.n.parent;
        }
      } else { 
        if (2*partRadius > (r * bhAngle)) { 
          // Open node, move on to first child
          index = curNode->u.n.nextNode;
	  
        } else {
          // Accept node, add to interact list, move on to sibling
	  xInteract->push_back(curNode->u.n.partCenter[0]);
	  yInteract->push_back(curNode->u.n.partCenter[1]);
	  zInteract->push_back(curNode->u.n.partCenter[2]);
	  mInteract->push_back(curNode->u.n.partMass);

          index = curNode->u.n.sibling;
 
          // If there is no sibling go up a level until we find a node
          ID_T parent = curNode->u.n.parent;
          while (index == -1 && parent != -1 && parent != root) {
            index = this->fNode[parent-nodeOffset].u.n.sibling;
            parent = this->fNode[parent-nodeOffset].u.n.parent;
          }
        }
      }
    }
  }

  // Force calculation for this particle
  this->fParticle[p].force = 
    forceCalculationFast(p, xInteract, yInteract, zInteract, mInteract);

  delete xInteract;
  delete yInteract;
  delete zInteract;
  delete mInteract;
}


void BHForceTree::treeForceGadgetTopDownFast2(
                    ID_T p,               // Particle to calculate force on
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius,  // Accept or ignore node not opened
		    vector<POSVEL_T>* xInteract,
		    vector<POSVEL_T>* yInteract,
		    vector<POSVEL_T>* zInteract,
		    vector<POSVEL_T>* mInteract,
		    double *timeWalk,
		    double *timeEval)
{
  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;
  POSVEL_T pos_x = this->xx[p];
  POSVEL_T pos_y = this->yy[p];
  POSVEL_T pos_z = this->zz[p];
  
  // Follow thread through tree from root choosing nodes and particles
  // which will contribute to the force of the given particle
  ID_T root = this->particleCount;
  ID_T index = root;

  clock_t start, end;

  start = clock();
  while (index >= 0) {

    if (index < this->nodeOffset) {
      // Particle
      dx = this->xx[index] - pos_x;
      dy = this->yy[index] - pos_y;
      dz = this->zz[index] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      if (r < critRadius && p != index) {
	xInteract->push_back(this->xx[index]);
	yInteract->push_back(this->yy[index]);
	zInteract->push_back(this->zz[index]);
	mInteract->push_back(this->mass[index]);
      }

      index = this->fParticle[index].nextNode;
    } else {
      // Node
      FNode* curNode = &this->fNode[index - this->nodeOffset];
      partRadius = curNode->u.n.partRadius;
      distToNearPoint = distanceToNearestPoint(pos_x, pos_y, pos_z, curNode);
      
      dx = curNode->u.n.partCenter[0] - pos_x;
      dy = curNode->u.n.partCenter[1] - pos_y;
      dz = curNode->u.n.partCenter[2] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);
      
      // Node is ignored if it is too far away from the particle
      // Distance from particle to particle radius exceeds critical radius
      // Distance from particle to nearest side of node exceeds critical radius
      if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
	
        // Ignore node, move on to sibling of this node
        index = curNode->u.n.sibling;
	
        // If there is no sibling go up a level until we find a node
        ID_T parent = curNode->u.n.parent;
        while (index == -1 && parent != -1 && parent != root) {
          index = this->fNode[parent - this->nodeOffset].u.n.sibling;
          parent = this->fNode[parent - this->nodeOffset].u.n.parent;
        }
      } else { 
        if (2*partRadius > (r * bhAngle)) { 
          // Open node, move on to first child
          index = curNode->u.n.nextNode;
	  
        } else {
          // Accept node, add to interact list, move on to sibling
	  xInteract->push_back(curNode->u.n.partCenter[0]);
	  yInteract->push_back(curNode->u.n.partCenter[1]);
	  zInteract->push_back(curNode->u.n.partCenter[2]);
	  mInteract->push_back(curNode->u.n.partMass);

          index = curNode->u.n.sibling;
 
          // If there is no sibling go up a level until we find a node
          ID_T parent = curNode->u.n.parent;
          while (index == -1 && parent != -1 && parent != root) {
            index = this->fNode[parent-nodeOffset].u.n.sibling;
            parent = this->fNode[parent-nodeOffset].u.n.parent;
          }
        }
      }
    }
  }
  end = clock();
  *timeWalk = 1.0*(end-start)/CLOCKS_PER_SEC;

  // Force calculation for this particle
  start = clock();
  this->fParticle[p].force = 
    forceCalculationFast(p, xInteract, yInteract, zInteract, mInteract);
  end = clock();
  *timeEval = 1.0*(end-start)/CLOCKS_PER_SEC;

  xInteract->clear();
  yInteract->clear();
  zInteract->clear();
  mInteract->clear();
}

/////////////////////////////////////////////////////////////////////////
//
// Short range gravity calculation for a single particle
// Starting with the particle walk up the parents processing siblings
// by testing particles and by opening, accepting or ignoring nodes.
// Stop moving up parents when the nearest side is beyond critical radius.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::treeForceGadgetBottomUp(
                    ID_T p,               // Particle to calculate force on
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius)  // Accept or ignore node not opened
{
  // Collect into interaction lists
  vector<ID_T>* partInteract = new vector<ID_T>;
  vector<ID_T>* nodeInteract = new vector<ID_T>;

  // Location of particle
  POSVEL_T dx, dy, dz, r, partRadius;
  POSVEL_T pos_x = this->xx[p];
  POSVEL_T pos_y = this->yy[p];
  POSVEL_T pos_z = this->zz[p];

  ID_T curId = p;
  ID_T parent = this->fParticle[curId].parent;

  while (parent != -1) {
    ID_T child = this->fNode[parent - this->nodeOffset].u.n.nextNode;
    while (child != -1) {
      if (child != curId) {
        if (child < this->nodeOffset) {
          // Particle
          dx = this->xx[child] - pos_x;
          dy = this->yy[child] - pos_y;
          dz = this->zz[child] - pos_z;
          r = sqrt(dx * dx + dy * dy + dz * dz);

          if (r < critRadius) {
            partInteract->push_back(child);
          }
          child = this->fParticle[child].sibling;
        }
        else {
          // Node
          FNode* childNode = &this->fNode[child - this->nodeOffset];
          partRadius = childNode->u.n.partRadius;
          POSVEL_T distToNearPoint = 
            distanceToNearestPoint(pos_x, pos_y, pos_z, childNode);

          dx = childNode->u.n.partCenter[0] - pos_x;
          dy = childNode->u.n.partCenter[1] - pos_y;
          dz = childNode->u.n.partCenter[2] - pos_z;
          r = sqrt(dx * dx + dy * dy + dz * dz);

          // Check for ignore of node first
          if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
            // Ignore
          } else

          if (2*partRadius < (r * bhAngle)) { 
            // Accept
            nodeInteract->push_back(child);

          } else {
            // Open node
            recurseOpenNode(childNode, pos_x, pos_y, pos_z, 
                            bhAngle, critRadius,
                            partInteract, nodeInteract);
          }
          child = this->fNode[child - this->nodeOffset].u.n.sibling;
        }
      }
      else {
        if (curId < this->nodeOffset)
          child = this->fParticle[curId].sibling;
        else
          child = this->fNode[child - this->nodeOffset].u.n.sibling;
      }
    }
    curId = parent;
    parent = this->fNode[parent - this->nodeOffset].u.n.parent;
  }

  // Force calculation for this particle
  this->fParticle[p].force = 
    forceCalculation(p, partInteract, nodeInteract);

  delete partInteract;
  delete nodeInteract;
}

/////////////////////////////////////////////////////////////////////////
//
// Open this node recursively adding accepted nodes and particles
// to the interact list
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::recurseOpenNode(
                    FNode* curNode,
                    POSVEL_T pos_x,
                    POSVEL_T pos_y,
                    POSVEL_T pos_z,
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius,  // Accept or ignore node not opened
                    vector<ID_T>* partInteract,
                    vector<ID_T>* nodeInteract)
{
  POSVEL_T dx, dy, dz, r, partRadius;
  ID_T child = curNode->u.n.nextNode;

  while (child != -1) {
    if (child < this->nodeOffset) {
      // Particle
      dx = this->xx[child] - pos_x;
      dy = this->yy[child] - pos_y;
      dz = this->zz[child] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      if (r < critRadius) {
        partInteract->push_back(child);
      }
      child = this->fParticle[child].sibling;
    } else {
      FNode* childNode = &this->fNode[child - this->nodeOffset];
      partRadius = childNode->u.n.partRadius;
      POSVEL_T distToNearPoint = 
        distanceToNearestPoint(pos_x, pos_y, pos_z, childNode);

      dx = childNode->u.n.partCenter[0] - pos_x;
      dy = childNode->u.n.partCenter[1] - pos_y;
      dz = childNode->u.n.partCenter[2] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      // Check for ignore of node first
      if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
        // Ignore
      } else

      if (2*partRadius < (r * bhAngle)) { 
        // Accept
        nodeInteract->push_back(child);

      } else {
        // Open node
        recurseOpenNode(childNode, pos_x, pos_y, pos_z, 
                        bhAngle, critRadius,
                        partInteract, nodeInteract);
      }
      child = this->fNode[child - this->nodeOffset].u.n.sibling;
    }
  }
} 

/////////////////////////////////////////////////////////////////////////
//
// Short range gravity calculation for every particle in the tree
// Recurses through the tree saving previous work for reuse when popping
// out of recursion.  Based on Barnes treecode.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::treeForceBarnesAdjust(
                    POSVEL_T bhAngle,     // Open node to examine children
                    POSVEL_T critRadius)  // Accept or ignore node not opened
{
  ID_T root = this->particleCount;
  vector<ID_T>* active = new vector<ID_T>;
  vector<ID_T>* partInteract = new vector<ID_T>;
  vector<ID_T>* nodeInteract = new vector<ID_T>;

  active->push_back(root);

  // Walk uses opening angle, critical radius for open, accept and ignore
  walkTreeBarnesAdjust(active, partInteract, nodeInteract, 
                       root, bhAngle, critRadius);
  
  delete active;
  delete partInteract;
  delete nodeInteract;
}

///////////////////////////////////////////////////////////////////////////
//
// Walk the BH tree for the given particle or node (identifier curId)
// Recursion starts with a new active list which will contain particles
// and nodes which possibly will contribute to the force on a particle.
// Particles on the active list will always be chosen for the interact list.
// Nodes on the active list may be OPENED if they are close enough
// or ACCEPTED and used in summary if they are within a critical radius
// and IGNORED otherwise. Nodes that are opened 
// have all their children (particles or nodes) added to the active list.
//
// After the children are added a new level of recursion starts by
// calculating a new size for that level, starting a fresh active list
// and building on the current interact lists.
//
// Recursion continues until the active list has been completely processed.
// When a level of recursion is complete the active list is destroyed
// and new items put on the interact lists are popped off.
//
// The advantage to this method is that items in the interaction list may
// not need to be processed again when we are doing the low levels of the tree.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::walkTreeBarnesAdjust(
                    vector<ID_T>* curActive,      // nodes to be acted on
                    vector<ID_T>* partInteract,   // particles for force
                    vector<ID_T>* nodeInteract,   // nodes for force
                    ID_T curId,                   // current particle or node
                    POSVEL_T bhAngle,             // open node
                    POSVEL_T critRadius)          // accept or ignore node
{
  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;

  // Current active list
  int begIndx = 0;
  int endIndx = curActive->size();

  // Construct active list for each recursion
  vector<ID_T>* newActive = new vector<ID_T>;

  // Set the location for the particle or node for the walk
  POSVEL_T pos_x, pos_y, pos_z;
  if (curId < this->nodeOffset) {
    pos_x = this->xx[curId];
    pos_y = this->yy[curId];
    pos_z = this->zz[curId];
  } else {
    FNode* curNode = &this->fNode[curId - this->nodeOffset];
    pos_x = curNode->u.n.partCenter[0];
    pos_y = curNode->u.n.partCenter[1];
    pos_z = curNode->u.n.partCenter[2];
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // Process the active list window adding children to end of list
  // Valid particles and accepted nodes are copied to the interact list
  //
  int hasChildren = 0;
  int pcount = 0;
  int ncount = 0;
  for (int indx = begIndx; indx < endIndx; indx++) {

    // If the current active element is a cell it will be
    // ACCEPTED and copied to the interact list
    // OPENED and its children will be added to the end of the active list
    // IGNORED because it is too far away
    if ((*curActive)[indx] >= this->nodeOffset) {
      hasChildren = 1;

      FNode* actNode = &this->fNode[(*curActive)[indx] - this->nodeOffset];
      partRadius = actNode->u.n.partRadius;
      distToNearPoint = distanceToNearestPoint(pos_x, pos_y, pos_z, actNode);

      dx = actNode->u.n.partCenter[0] - pos_x;
      dy = actNode->u.n.partCenter[1] - pos_y;
      dz = actNode->u.n.partCenter[2] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      // Node is ignored if it is too far away from the particle
      // Distance from particle to particle radius exceeds critical radius
      // Distance from particle to nearest side of node exceeds critical radius
      if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {

        // Ignore node, move on to sibling of this node
      }
      else {
        if (2*partRadius > (r * bhAngle)) {
          // Open node, move on to first child
          ID_T child =
            this->fNode[(*curActive)[indx] - this->nodeOffset].u.n.nextNode;
          while (child != -1) {
            if (child >= this->nodeOffset) {

              // Child is a node which is active and must be considered
              newActive->push_back(child);
              child = this->fNode[child - this->nodeOffset].u.n.sibling;
            }
            else {
              // Child is a particle, add to interaction list
              partInteract->push_back(child);
              pcount++;
              child = this->fParticle[child].sibling;
            }
          }
        } else {
          // Accept node, add to interact list, move on to sibling
          nodeInteract->push_back((*curActive)[indx]);
          ncount++;
        }
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // At this point a new level of children may have been added to active list
  // Process children by dividing the node size and recursing
  //
  if (hasChildren) {

    // Current item on active list is a cell
    if (curId >= this->nodeOffset) {

      // Process each child
      ID_T child = fNode[curId - this->nodeOffset].u.n.nextNode;
      while (child != -1) {

        // Child is a node
        if (child >= this->nodeOffset) {
          FNode* childNode = &this->fNode[child - this->nodeOffset];

          // Recurse on walk tree to process child
          walkTreeBarnesAdjust(newActive, partInteract, nodeInteract,
                               child, bhAngle, critRadius);
          child = childNode->u.n.sibling;
        }
        // Child is a particle
        else {
           walkTreeBarnesAdjust(newActive, partInteract, nodeInteract,
                                child, bhAngle, critRadius);
           child = this->fParticle[child].sibling;
        }
      }
    }
    // Current item on active list is a particle
    else {
      walkTreeBarnesAdjust(newActive, partInteract, nodeInteract,
                           curId, bhAngle, critRadius);
    }
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // If no new items were added to active list we are done and can process
  // the interact list for this particle p (which can't be a cell)
  //
  else {
    if (curId > this->nodeOffset)
      cout << "ERROR: POP OUT ON NODE " << curId << endl;

    // Since the interact lists might contain accepted nodes from upper levels
    // which need to be opened for this particle, adjust the lists first
    vector<ID_T>* adjNodeInteract = new vector<ID_T>;
    vector<ID_T>* adjPartInteract = new vector<ID_T>;

    static Timings::TimerRef adjtimer = Timings::getTimer("Barnes Adjustment");
    Timings::startTimer(adjtimer);
    adjustInteraction(curId,
                      partInteract, nodeInteract,
                      adjPartInteract, adjNodeInteract,
                      bhAngle, critRadius);
    Timings::stopTimer(adjtimer);

    // Calculate force for the particle
    this->fParticle[curId].force = 
      forceCalculation(curId, adjPartInteract, adjNodeInteract);

    delete adjNodeInteract;
    delete adjPartInteract;
  }

  // Active list is new for every recursion level
  // Interact lists are appended to at each recursion level
  // So interact lists must be popped by the correct number for this recursion
  for (int i = 0; i < pcount; i++)
    partInteract->pop_back();
  for (int i = 0; i < ncount; i++)
    nodeInteract->pop_back();
  delete newActive;
}


/////////////////////////////////////////////////////////////////////////
//
// Recursion enters with a guess for the interact lists which were set
// on previous invocations of the method.  Check the node interact list
// which contains nodes which were accepted to see if they should 
// actually be opened relative to this new current particle or node
// If so remove from the nodeInteract list and add to the active list
//
// Particles in the interact list might actually be grouped and used
// with their parent node as an accept, but leaving them will lead to
// a better answer, not a worse.  So we won't change the partInteract
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::adjustInteraction(
                    ID_T p0,
                    vector<ID_T>* partInteract,
                    vector<ID_T>* nodeInteract,
                    vector<ID_T>* adjPartInteract,
                    vector<ID_T>* adjNodeInteract,
                    POSVEL_T bhAngle,
                    POSVEL_T critRadius)
{
  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;

  // Get location of particle being adjusted for
  POSVEL_T pos_x = this->xx[p0];
  POSVEL_T pos_y = this->yy[p0];
  POSVEL_T pos_z = this->zz[p0];

  // Copy all particles to the adjust list, will only add new particles
  int numberOfParticles = (int) partInteract->size();
  for (int p = 0; p < numberOfParticles; p++)
    adjPartInteract->push_back((*partInteract)[p]);

  // Process each node to see if status changes from accept to ignore or open
  int numberOfNodes = (int) nodeInteract->size();
  for (int n = 0; n < numberOfNodes; n++) {
    FNode* curNode = &this->fNode[(*nodeInteract)[n] - this->nodeOffset];
    partRadius = curNode->u.n.partRadius;
    distToNearPoint = distanceToNearestPoint(pos_x, pos_y, pos_z, curNode);

    dx = curNode->u.n.partCenter[0] - pos_x;
    dy = curNode->u.n.partCenter[1] - pos_y;
    dz = curNode->u.n.partCenter[2] - pos_z;
    r = sqrt(dx * dx + dy * dy + dz * dz);

    // Node is ignored if it is too far away from the particle
    // Distance from particle to particle radius exceeds critical radius
    // Distance from particle to nearest side of node exceeds critical radius
    if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
        // Ignore node, move on to sibling of this node
      }
    else {
      if (2*partRadius > (r * bhAngle)) {
        // Node must be opened and constituent parts examined
        adjustNodeInteract(p0, curNode, adjPartInteract, adjNodeInteract,
                           bhAngle, critRadius);
      } else {
        // Accept node, add to interact list, move on to sibling
        adjNodeInteract->push_back((*nodeInteract)[n]);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Recursive part of interaction adjustment
// Examine children of current node recursively for inclusion into interaction
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::adjustNodeInteract(
                    ID_T p0,
                    FNode* curNode,
                    vector<ID_T>* adjPartInteract,
                    vector<ID_T>* adjNodeInteract,
                    POSVEL_T bhAngle,
                    POSVEL_T critRadius)
{
  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;

  // Get location of particle being adjusted for
  POSVEL_T pos_x = this->xx[p0];
  POSVEL_T pos_y = this->yy[p0];
  POSVEL_T pos_z = this->zz[p0];

  // Current node is to be opened and recursively checked for interactions
  ID_T child = curNode->u.n.nextNode;
  while (child != -1) {
    if (child < this->nodeOffset) {
      // Child is a particle, add to adjusted particle interact list
      adjPartInteract->push_back(child);
      child = this->fParticle[child].sibling;
    }
    else {
      // Child is a node, check to see if it should be opened, accepted, ignored
      FNode* childNode = &this->fNode[child - this->nodeOffset];
      partRadius = childNode->u.n.partRadius;
      distToNearPoint = distanceToNearestPoint(pos_x, pos_y, pos_z, childNode);

      dx = childNode->u.n.partCenter[0] - pos_x;
      dy = childNode->u.n.partCenter[1] - pos_y;
      dz = childNode->u.n.partCenter[2] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      // Node is ignored if it is too far away from the particle
      // Distance from particle to particle radius exceeds critical radius
      // Distance from particle to nearest side of node exceeds critical radius
      if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
          // Ignore node, move on to sibling of this node
      }
      else {
        if (2*partRadius > (r * bhAngle)) {
          // Node must be opened and constituent parts examined
          adjustNodeInteract(p0, childNode, adjPartInteract, adjNodeInteract,
                             bhAngle, critRadius);
        } else {
          // Accept node
          adjNodeInteract->push_back(child);
        }
      }
      child = this->fNode[child - this->nodeOffset].u.n.sibling;
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Short range gravity calculation for every particle in the tree
// Recurses through the tree saving previous work for reuse when popping
// out of recursion.  Based on Barnes treecode with quick scan.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::treeForceBarnesQuick(
                    POSVEL_T bhAngle,     // Open a node
                    POSVEL_T critRadius)  // Accept or ignore node not opened
{
  ID_T root = this->particleCount;

  vector<ID_T>* active = new vector<ID_T>;
  vector<ID_T>* partInteract = new vector<ID_T>;
  vector<ID_T>* nodeInteract = new vector<ID_T>;

  active->push_back(root);

  // Quick walk of tree accepts nodes that do not touch target node
  walkTreeBarnesQuick(active, partInteract, nodeInteract, 
                      root, bhAngle, critRadius);

  delete active;
  delete partInteract;
  delete nodeInteract;
}

/////////////////////////////////////////////////////////////////////////
//
// Walk the BH tree for the given particle or node (identifier curId)
// Recursion starts with a new active list which will contain particles
// and nodes which possibly will contribute to the force on a particle.
// Particles on the active list will always be chosen for the interact list.
// Nodes on the active list may be OPENED if they are close enough
// or ACCEPTED and used in summary if they are not. Nodes that are opened 
// have all their children (particles or nodes) added to the active list.
//
// After the children are added a new level of recursion starts by
// calculating a new size for that level, starting a fresh active list
// and building on the current interact lists.
//
// Recursion continues until the active list has been completely processed.
// When a level of recursion is complete the active list is destroyed
// and new items put on the interact lists are popped off.
//
// The advantage to this method is that items in the interaction list may
// not need to be processed again when we are doing the low levels of the tree.
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::walkTreeBarnesQuick(
                    vector<ID_T>* curActive,      // nodes to be acted on
                    vector<ID_T>* partInteract,   // particles for force
                    vector<ID_T>* nodeInteract,   // nodes for force
                    ID_T curId,                   // current particle or node
                    POSVEL_T bhAngle,             // open node
                    POSVEL_T critRadius)          // accept or ignore
{
  POSVEL_T dx, dy, dz, r, partRadius, distToNearPoint;

  // Current active list
  int begIndx = 0;
  int endIndx = curActive->size();

  // Construct active list for each recursion
  vector<ID_T>* newActive = new vector<ID_T>;

  // Set the location for the particle or node for the walk
  POSVEL_T pos_x, pos_y, pos_z;
  if (curId < this->nodeOffset) {
    pos_x = this->xx[curId];
    pos_y = this->yy[curId];
    pos_z = this->zz[curId];
  } else {
    FNode* curNode = &this->fNode[curId - this->nodeOffset];
    pos_x = curNode->u.n.partCenter[0];
    pos_y = curNode->u.n.partCenter[1];
    pos_z = curNode->u.n.partCenter[2];
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // Process the active list window adding children to end of list
  // Valid particles and accepted nodes are copied to the interact list
  //
  int hasChildren = 0;
  int pcount = 0;
  int ncount = 0;
  for (int indx = begIndx; indx < endIndx; indx++) {

    // If the current active element is a cell it will be
    // ACCEPTED and copied to the interact list
    // OPENED and its children will be added to the end of the active list
    // IGNORED because it is too far away
    if ((*curActive)[indx] >= this->nodeOffset) {
      hasChildren = 1;

      FNode* actNode = &this->fNode[(*curActive)[indx] - this->nodeOffset];
      partRadius = actNode->u.n.partRadius;
      distToNearPoint = distanceToNearestPoint(pos_x, pos_y, pos_z, actNode);

      dx = actNode->u.n.partCenter[0] - pos_x;
      dy = actNode->u.n.partCenter[1] - pos_y;
      dz = actNode->u.n.partCenter[2] - pos_z;
      r = sqrt(dx * dx + dy * dy + dz * dz);

      // Node is ignored if it is too far away from the particle
      // Distance from particle to particle radius exceeds critical radius
      // Distance from particle to nearest side of node exceeds critical radius
      if ((r - partRadius) > critRadius || distToNearPoint > critRadius) {
        // Ignore node, move on to sibling of this node
      }
      else {
        if (2*partRadius > (r * bhAngle)) {
          // Open node, move on to first child
          ID_T child = 
            this->fNode[(*curActive)[indx] - this->nodeOffset].u.n.nextNode;
          while (child != -1) {
            if (child >= this->nodeOffset) {

              // Child is a node which is active and must be considered
              newActive->push_back(child);
              child = this->fNode[child - this->nodeOffset].u.n.sibling;
            }
            else {
              // Child is a particle, add to interaction list
              partInteract->push_back(child);
              pcount++;
              child = this->fParticle[child].sibling;
            }
          }
        } else {
          // Accept node, add to interact list, move on to sibling
          nodeInteract->push_back((*curActive)[indx]);
          ncount++;
        }
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // At this point a new level of children may have been added to active list
  // Process children by dividing the node size and recursing
  //
  if (hasChildren) {

    // Current item on active list is a cell
    if (curId >= this->nodeOffset) {

      // Process each child
      ID_T child = fNode[curId - this->nodeOffset].u.n.nextNode;
      while (child != -1) {

        // Child is a node
        if (child >= this->nodeOffset) {
          FNode* childNode = &this->fNode[child - this->nodeOffset];

          // Recurse on walk tree to process child
          walkTreeBarnesQuick(newActive, partInteract, nodeInteract,
                              child, bhAngle, critRadius);
          child = childNode->u.n.sibling;
        }
        // Child is a particle
        else {
           walkTreeBarnesQuick(newActive, partInteract, nodeInteract,
                               child, bhAngle, critRadius);
           child = this->fParticle[child].sibling;
        }
      }
    }
    // Current item on active list is a particle
    else {
      walkTreeBarnesQuick(newActive, partInteract, nodeInteract,
                          curId, bhAngle, critRadius);
    }
  }

  /////////////////////////////////////////////////////////////////////////
  //
  // If no new items were added to active list we are done and can process
  // the interact list for this particle p (which can't be a cell)
  //
  else {
    if (curId > this->nodeOffset)
      cout << "ERROR: POP OUT ON NODE " << curId << endl;
    this->fParticle[curId].force = 
      forceCalculation(curId, partInteract, nodeInteract);
  }

  // Active list is new for every recursion level
  // Interact lists are appended to at each recursion level
  // So interact lists must be popped by the correct number for this recursion
  for (int i = 0; i < pcount; i++)
    partInteract->pop_back();
  for (int i = 0; i < ncount; i++)
    nodeInteract->pop_back();
  delete newActive;
}

/////////////////////////////////////////////////////////////////////////
//
// Short range force calculation
// Potential is calculated and is used to determine the acceleration of
// the particle.  Acceleration is applied to the current velocity to
// produce the velocity at the next time step.
//
/////////////////////////////////////////////////////////////////////////

POSVEL_T BHForceTree::forceCalculation(
                    ID_T p0,                    // Target particle index
                    vector<ID_T>* partInteract, // Particles acting on p
                    vector<ID_T>* nodeInteract) // Nodes acting on p
{
  POSVEL_T accel[DIMENSION];
  POSVEL_T phi = 0.0;

  POSVEL_T pos0_x = this->xx[p0];
  POSVEL_T pos0_y = this->yy[p0];
  POSVEL_T pos0_z = this->zz[p0];

  for (int dim = 0; dim < DIMENSION; dim++)
    accel[dim] = 0.0;

  int numberOfNodes = (int) nodeInteract->size();
  int numberOfParticles = (int) partInteract->size();

  // Particles contributing to the force use location and mass of one particle
  for (int p = 0; p < numberOfParticles; p++) {
    ID_T particle = (*partInteract)[p];
    if (p0 != particle) {
      POSVEL_T dx = this->xx[particle] - pos0_x;
      POSVEL_T dy = this->yy[particle] - pos0_y;
      POSVEL_T dz = this->zz[particle] - pos0_z;

      POSVEL_T r2 = dx * dx + dy * dy + dz * dz;

      POSVEL_T f_over_r = this->mass[particle] * m_fl->f_over_r(r2);
      //POSVEL_T f_over_r = this->mass[particle] / r2;
      phi -= f_over_r;
      //if (p0 == 171893) cout << "Top Particle used " << particle << " phi " << phi << endl;

      accel[0] += dx * f_over_r * m_fcoeff;
      accel[1] += dy * f_over_r * m_fcoeff;
      accel[2] += dz * f_over_r * m_fcoeff;

      this->vx[p0] += dx * f_over_r * m_fcoeff;
      this->vy[p0] += dy * f_over_r * m_fcoeff;
      this->vz[p0] += dz * f_over_r * m_fcoeff;
    }
  }

  // Nodes contributing to force use center of mass and total particle mass
  for (int n = 0; n < numberOfNodes; n++) {
    FNode* node = &this->fNode[(*nodeInteract)[n] - this->nodeOffset];
    POSVEL_T dx = node->u.n.partCenter[0] - pos0_x;
    POSVEL_T dy = node->u.n.partCenter[1] - pos0_y;
    POSVEL_T dz = node->u.n.partCenter[2] - pos0_z;

    POSVEL_T r2 = dx * dx + dy * dy + dz * dz;

    POSVEL_T f_over_r = node->u.n.partMass * m_fl->f_over_r(r2);
    //POSVEL_T f_over_r = node->u.n.partMass / r2;
    phi -= f_over_r;
    //if (p0 == 171893) cout << "Top node used " << (*nodeInteract)[n] << " phi " << phi << endl;

    accel[0] += dx * f_over_r * m_fcoeff;
    accel[1] += dy * f_over_r * m_fcoeff;
    accel[2] += dz * f_over_r * m_fcoeff;
    
    this->vx[p0] += dx * f_over_r * m_fcoeff;
    this->vy[p0] += dy * f_over_r * m_fcoeff;
    this->vz[p0] += dz * f_over_r * m_fcoeff;
  }
  return phi;
}

POSVEL_T BHForceTree::forceCalculationFast(
                    ID_T p0,                    // Target particle index
                    vector<POSVEL_T>* xInteract,
                    vector<POSVEL_T>* yInteract,
                    vector<POSVEL_T>* zInteract,
                    vector<POSVEL_T>* mInteract)
{
  POSVEL_T phi = 0.0;

  POSVEL_T pos0_x = this->xx[p0];
  POSVEL_T pos0_y = this->yy[p0];
  POSVEL_T pos0_z = this->zz[p0];

  int nInteract = (int) xInteract->size();

  for (int p = 0; p < nInteract; p++) {
    POSVEL_T dx = (*xInteract)[p] - pos0_x;
    POSVEL_T dy = (*yInteract)[p] - pos0_y;
    POSVEL_T dz = (*zInteract)[p] - pos0_z;

    POSVEL_T r2 = dx * dx + dy * dy + dz * dz;

    POSVEL_T f_over_r = (*mInteract)[p] * m_fl->f_over_r(r2);
    //POSVEL_T f_over_r = this->mass[particle] / r2;

    this->vx[p0] += dx * f_over_r * m_fcoeff;
    this->vy[p0] += dy * f_over_r * m_fcoeff;
    this->vz[p0] += dz * f_over_r * m_fcoeff;
  }

  return phi;
}

/////////////////////////////////////////////////////////////////////////
//
// Return the distance^2 from location to the closest point on FNode
//
/////////////////////////////////////////////////////////////////////////

POSVEL_T BHForceTree::distanceToNearestPoint(
                              POSVEL_T pos_x,
                              POSVEL_T pos_y,
                              POSVEL_T pos_z,
                              FNode* node)
{
  // Calculate bounding box of current node
  // Nearest point in bounding box decides whether particle or node is used
  POSVEL_T dx, dy, dz, r;
  POSVEL_T minBound[DIMENSION], maxBound[DIMENSION];
  for (int dim = 0; dim < DIMENSION; dim++) {
    minBound[dim] = node->geoCenter[dim] - (node->geoSide[dim] * 0.5);
    maxBound[dim] = node->geoCenter[dim] + (node->geoSide[dim] * 0.5);
  }

  if (pos_x < minBound[0])
    dx = minBound[0] - pos_x;
  else if (pos_x > maxBound[0])
    dx = pos_x - maxBound[0];
  else
    dx = 0.0;

  if (pos_y < minBound[1])
    dy = minBound[1] - pos_y;
  else if (pos_y > maxBound[1])
    dy = pos_y - maxBound[1];
  else
    dy = 0.0;

  if (pos_z < minBound[2])
    dz = minBound[2] - pos_z;
  else if (pos_z > maxBound[2])
    dz = pos_z - maxBound[2];
  else
    dz = 0.0;

  r = sqrt(dx * dx + dy * dy + dz * dz);
  return r;
}

/////////////////////////////////////////////////////////////////////////
//
// Return the distance from location to the fNode center of mass
//
/////////////////////////////////////////////////////////////////////////

POSVEL_T BHForceTree::distanceToCenterOfMass(
                              POSVEL_T xLoc,
                              POSVEL_T yLoc,
                              POSVEL_T zLoc,
                              FNode* node)
{
  POSVEL_T xdist = (POSVEL_T) fabs(xLoc - node->u.n.partCenter[0]);
  POSVEL_T ydist = (POSVEL_T) fabs(yLoc - node->u.n.partCenter[1]);
  POSVEL_T zdist = (POSVEL_T) fabs(zLoc - node->u.n.partCenter[2]);
  POSVEL_T dist = sqrt((xdist * xdist) + (ydist * ydist) + (zdist * zdist));
  return dist;
}

/////////////////////////////////////////////////////////////////////////
//
// Return the distance from location to the fNode furthest corner
//
/////////////////////////////////////////////////////////////////////////

POSVEL_T BHForceTree::distanceToFarCorner(
                              POSVEL_T xLoc,
                              POSVEL_T yLoc,
                              POSVEL_T zLoc,
                              FNode* node)
{
  POSVEL_T distance = 0.0;
  POSVEL_T corner[DIMENSION];
  POSVEL_T xdist, ydist, zdist, dist;

  for (int k = -1; k <= 1; k=k+2) {
    corner[2] = node->geoCenter[2] + (k * (node->geoSide[2] * 0.5));
    for (int j = -1; j <= 1; j=j+2) {
      corner[1] = node->geoCenter[1] + (j * (node->geoSide[1] * 0.5));
      for (int i = -1; i <= 1; i=i+2) {
        corner[0] = node->geoCenter[0] + (i * (node->geoSide[0] * 0.5));

        xdist = (POSVEL_T) fabs(xLoc - corner[0]);
        ydist = (POSVEL_T) fabs(yLoc - corner[1]);
        zdist = (POSVEL_T) fabs(zLoc - corner[2]);
        dist = sqrt((xdist * xdist) + (ydist * ydist) + (zdist * zdist));
        if (dist > distance)
          distance = dist;
      }
    }
  }
  return distance;
}

/////////////////////////////////////////////////////////////////////////
//
// Return the distance from location to the fNode nearest corner
//
/////////////////////////////////////////////////////////////////////////

POSVEL_T BHForceTree::distanceToNearCorner(
                              POSVEL_T xLoc,
                              POSVEL_T yLoc,
                              POSVEL_T zLoc,
                              FNode* node)
{
  POSVEL_T distance = MAX_FLOAT;
  POSVEL_T corner[DIMENSION];
  POSVEL_T xdist, ydist, zdist, dist;

  for (int k = -1; k <= 1; k=k+2) {
    corner[2] = node->geoCenter[2] + (k * (node->geoSide[2] * 0.5));
    for (int j = -1; j <= 1; j=j+2) {
      corner[1] = node->geoCenter[1] + (j * (node->geoSide[1] * 0.5));
      for (int i = -1; i <= 1; i=i+2) {
        corner[0] = node->geoCenter[0] + (i * (node->geoSide[0] * 0.5));

        xdist = (POSVEL_T) fabs(xLoc - corner[0]);
        ydist = (POSVEL_T) fabs(yLoc - corner[1]);
        zdist = (POSVEL_T) fabs(zLoc - corner[2]);
        dist = sqrt((xdist * xdist) + (ydist * ydist) + (zdist * zdist));
        if (dist < distance)
          distance = dist;
      }
    }
  }
  return distance;
}

/////////////////////////////////////////////////////////////////////////
//
// Print BH tree with indentations indicating levels
// Since the tree has been threaded changing the recursive tree with children
// into an iterative tree with next nodes and parents, walk the tree
// iteratively keeping track of parents to indicate when levels change
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::printBHForceTree()
{
  ID_T curIndex = this->nodeOffset;
  vector<ID_T> parents;
  parents.push_back(-1);
  ID_T parentIndex = 0;

  while (curIndex != -1) {

    // Get the parent of the current index
    ID_T parent;
    if (curIndex >= this->nodeOffset)
      parent = this->fNode[curIndex - this->nodeOffset].u.n.parent;
    else
      parent = this->fParticle[curIndex].parent;

    // Pop the stack of parents until the level is right
    while (parent != parents[parentIndex]) {
      parents.pop_back();
      parentIndex--;
    }

    // Print FNode
    if (curIndex >= this->nodeOffset) {
      FNode* fn = &this->fNode[curIndex-this->nodeOffset];

      cout << parentIndex << ":" << setw(parentIndex) << " ";
      cout << "N " << curIndex 
           << " sibling " << fn->u.n.sibling 
           << " next " << fn->u.n.nextNode 
           << " parent " << fn->u.n.parent 

           << " [" << (fn->geoCenter[0]-fn->geoSide[0]/2.0)
           << ":" << (fn->geoCenter[0]+fn->geoSide[0]/2.0) << "] "
           << " [" << (fn->geoCenter[1]-fn->geoSide[1]/2.0)
           << ":" << (fn->geoCenter[1]+fn->geoSide[1]/2.0) << "] "
           << " [" << (fn->geoCenter[2]-fn->geoSide[2]/2.0)
           << ":" << (fn->geoCenter[2]+fn->geoSide[2]/2.0) << "] "

           << " (" << fn->u.n.partCenter[0] 
           << " ," << fn->u.n.partCenter[1] 
           << " ," << fn->u.n.partCenter[2]

           << ") MASS " << fn->u.n.partMass
           << " RADIUS " << fn->u.n.partRadius
           << endl;
        
      // Push back the new FNode which will have children
      parents.push_back(curIndex);
      parentIndex++;

      // Walk to next node (either particle or node)
      curIndex = this->fNode[curIndex-this->nodeOffset].u.n.nextNode;
    }

    // Print FParticle
    else {
      cout << parentIndex << ":" << setw(parentIndex) << " ";
      cout << "P " << curIndex 
           << " sibling " << this->fParticle[curIndex].sibling 
           << " next " << this->fParticle[curIndex].nextNode 
           << " parent " << this->fParticle[curIndex].parent
           << " (" << xx[curIndex]
           << " ," << yy[curIndex]
           << " ," << zz[curIndex] << ")" << endl;

      // Walk to next node (either particle or node)
      curIndex = this->fParticle[curIndex].nextNode;
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Print the force values for comparison
//
/////////////////////////////////////////////////////////////////////////

void BHForceTree::printForceValues()
{
  for (int p = 0; p < this->particleCount; p++) {
    cout << "Particle: " << setw(8) << p
         << " force " << this->fParticle[p].force << endl;
  }
}

/////////////////////////////////////////////////////////////////////////
//
// Get the index of the child which should contain this particle
//
/////////////////////////////////////////////////////////////////////////

int BHForceTree::getChildIndex(FNode* node, ID_T pindx)
{
  // Vary Z dimension fastest in making octtree children
  int index = 0;
  if (this->xx[pindx] >= node->geoCenter[0]) index += 4;
  if (this->yy[pindx] >= node->geoCenter[1]) index += 2;
  if (this->zz[pindx] >= node->geoCenter[2]) index += 1;
  return index;
}
