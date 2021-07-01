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

// .NAME BHForceTree - Create a Barnes Hut tree from the given particles
//
// .SECTION Description
// BHTree takes particle locations and distributes them recursively in
// a Barnes Hut tree.  The tree is an octree, dividing on the physical
// location such that one particle or one node appears within a child
// so that it is essentially AMR for particles.
//
// After the tree is created it is walked using depth first recursion and
// the nodes are threaded together so that the tree becomes iterative.
// By stringing nodes together rather than maintaining indices into children
// summary information for each node can replace the 8 integer slots that
// were taken up by the children.  Now each node can maintain the mass
// below, the length of the physical box it represents and the center of
// mass of particles within the node.
//
// Each particle and each node maintains an index for the next node and
// also the parent, so that it is possible to represent the recursive tree
// by paying attention to parents.
//

#ifndef BHForceTree_h
#define BHForceTree_h

#include "BasicDefinition.h"
#include "bigchunk.h"

#include <string>
#include <vector>
#include <algorithm>

#include "ForceLaw.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////
//
// Force Particles
//
/////////////////////////////////////////////////////////////////////////

class FParticle {
public:
  FParticle();

  ID_T      sibling;		// Next on same level, ending in -1
  ID_T      nextNode;		// Next node in iteration, particle or node
  ID_T      parent;		// Parent FNode
  POSVEL_T  force;
};

/////////////////////////////////////////////////////////////////////////
//
// Force Nodes
//
// Barnes Hut octree structure for N-body is represented by vector
// of FNodes which divide space into octants which are filled with one
// particle or one branching node.  As the tree is built the child[8]
// array is used.  Afterwards the tree is walked linking the nodes and
// replacing the child structure with data about the tree.  When building
// the tree child information is an integer which is the index of the
// halo particle which was put into a vector of FParticle, or the index
// of the FNode offset by the number of particles
//
/////////////////////////////////////////////////////////////////////////

class FNode {
public:
  FNode(POSVEL_T* minLoc, POSVEL_T* maxLoc);
  FNode(FNode* parent, int child);

  POSVEL_T geoSide[DIMENSION];		// Length of octant on each side
  POSVEL_T geoCenter[DIMENSION];	// Physical center of octant

  union {
    ID_T  child[NUM_CHILDREN];		// Index of particle or node
    struct NodeInfo {
      POSVEL_T partCenter[DIMENSION];
      POSVEL_T partMass;
      POSVEL_T partRadius;
      ID_T     sibling;
      ID_T     nextNode;
      ID_T     parent;
    } n;
  } u;
};

/////////////////////////////////////////////////////////////////////////
//
// Barnes Hut octree of FParticles and FNodes threaded
//
/////////////////////////////////////////////////////////////////////////

class BHForceTree {
public:
  BHForceTree(
	      POSVEL_T* minLoc,       // Bounding box of halo
	      POSVEL_T* maxLoc,       // Bounding box of halo
	      ID_T count,             // Number of particles in halo
	      POSVEL_T* xLoc,         // Locations of every particle
	      POSVEL_T* yLoc,
	      POSVEL_T* zLoc,
	      POSVEL_T* xVel,         // Velocities of every particle
	      POSVEL_T* yVel,
	      POSVEL_T* zVel,
	      POSVEL_T* mass,	      // Mass of each particle
	      POSVEL_T avgMass);      // Average mass for estimation

  BHForceTree(
	      POSVEL_T* minLoc,       // Bounding box of halo
	      POSVEL_T* maxLoc,       // Bounding box of halo
	      ID_T count,             // Number of particles in halo
	      POSVEL_T* xLoc,         // Locations of every particle
	      POSVEL_T* yLoc,
	      POSVEL_T* zLoc,
	      POSVEL_T* xVel,         // Velocities of every particle
	      POSVEL_T* yVel,
	      POSVEL_T* zVel,
	      POSVEL_T* mass,	      // Mass of each particle
	      POSVEL_T avgMass,       // Average mass for estimation
	      ForceLaw *fl,
	      float fcoeff);

  ~BHForceTree();

  ////////////////////////////////////////////////////////////////////////////
  //
  // Create the BH tree recursively by placing particles in empty octants
  //
  void createBHForceTree();

  ////////////////////////////////////////////////////////////////////////////
  //
  // Walk tree to thead so that it can be accessed iteratively or recursively
  //
  void threadBHForceTree(
	ID_T curIndx,           // Current node/particle
	ID_T sibling,           // Sibling of current
	ID_T parent,            // Parent of current
	ID_T* lastIndx,	        // Last node/particle
	POSVEL_T* radius);      // Needed to pass up particle radius of child

  POSVEL_T distanceToCenterOfMass(
	POSVEL_T xLoc,		// Distance from point to node particle center
	POSVEL_T yLoc,
	POSVEL_T zLoc,
	FNode* node);

  POSVEL_T distanceToNearCorner(
	POSVEL_T xLoc,		// Distance from point to node nearest corner
	POSVEL_T yLoc,
	POSVEL_T zLoc,
	FNode* node);

  POSVEL_T distanceToFarCorner(
	POSVEL_T xLoc,		// Distance from point to node furthest corner
	POSVEL_T yLoc,
	POSVEL_T zLoc,
	FNode* node);

  POSVEL_T distanceToNearestPoint(
	POSVEL_T xLoc,		// Distance from point to node closest point
	POSVEL_T yLoc,
	POSVEL_T zLoc,
	FNode* node);

  ////////////////////////////////////////////////////////////////////////////
  //
  // Calculate force using N^2 method
  //
  void treeForceN2(
	POSVEL_T critRadius);       // Criteria for ignoring a node

  ////////////////////////////////////////////////////////////////////////////
  //
  // Short range force calculation on all particles of the tree
  // Recurse through levels saving information for reuse
  // Based on Barnes treecode
  //
  void treeForceBarnesAdjust(
	POSVEL_T openAngle,         // Criteria for opening a node
	POSVEL_T critRadius);       // Criteria for ignoring a node

  // Walk tree using opening angle and critical radius
  void walkTreeBarnesAdjust(
	vector<ID_T>* active,	    // List of nodes which must be acted on
	vector<ID_T>* partInteract,  // Particles which act on object
	vector<ID_T>* nodeInteract,  // Nodes which act on object
	ID_T curId,                 // Id of current particle or node
	POSVEL_T bhAngle,           // Opening angle squared
	POSVEL_T critRadius);       // Critical radius squared

  // Barnes tree walk will accept nodes that should be opened because of
  // comparison between two nodes higher in the recursion.  Adust this
  // when calculating for a particular particle.
  void adjustInteraction(
	ID_T p0,
	vector<ID_T>* partInteract,
	vector<ID_T>* nodeInteract,
	vector<ID_T>* adjPartInteract,
	vector<ID_T>* adjNodeInteract,
	POSVEL_T bhAngle,
	POSVEL_T critRadius);

  // Recursive part of interaction adjustment
  void adjustNodeInteract(
	ID_T p0,
	FNode* curNode,
	vector<ID_T>* adjPartInteract,
	vector<ID_T>* adjNodeInteract,
	POSVEL_T bhAngle,
	POSVEL_T critRadius);

  ////////////////////////////////////////////////////////////////////////////
  //
  // Short range force calculation on all particles of the tree
  // Recurse through levels saving information for reuse
  // Based on Barnes treecode with quick scan where nodes are accepted
  // if they touch the target node.
  //
  void treeForceBarnesQuick(
	POSVEL_T openAngle,         // Criteria fo opening a node
	POSVEL_T critRadius);       // Criteria for ignoring a node

  // Walk tree opening only nodes that physically touch target node
  void walkTreeBarnesQuick(
	vector<ID_T>* active,	    // List of nodes which must be acted on
	vector<ID_T>* partInteract,  // Particles which act on object
	vector<ID_T>* nodeInteract,  // Nodes which act on object
	ID_T curId,                 // Id of current particle or node
	POSVEL_T bhAngle,           // Opening angle squared
	POSVEL_T critRadius);       // Critical radius squared

  ////////////////////////////////////////////////////////////////////////////
  //
  // Calculate force on individual particles using tree walks
  // Short range force on one particle starting from root walking down
  //
  void treeForceGadgetTopDown(
	ID_T p,                     // Index of particle for calculation
	POSVEL_T openAngle,         // Criteria for opening a node
	POSVEL_T critRadius);       // Criteria for ignoring a node

  void treeForceGadgetTopDownFast(
	ID_T p,                     // Index of particle for calculation
	POSVEL_T openAngle,         // Criteria for opening a node
	POSVEL_T critRadius);       // Criteria for ignoring a node

  void treeForceGadgetTopDownFast2(
	ID_T p,                     // Index of particle for calculation
	POSVEL_T openAngle,         // Criteria for opening a node
	POSVEL_T critRadius,        // Criteria for ignoring a node
	vector<POSVEL_T>* xInteract,
	vector<POSVEL_T>* yInteract,
	vector<POSVEL_T>* zInteract,
	vector<POSVEL_T>* mInteract,
	double *timeWalk,
	double *timeEval);

  ////////////////////////////////////////////////////////////////////////////
  //
  // Calculate force on individual particles using tree walks
  // Short range force on one particle starting with particle walking up
  //
  void treeForceGadgetBottomUp(
	ID_T p,                     // Index of particle for calculation
	POSVEL_T openAngle,         // Criteria for opening a node
	POSVEL_T critRadius);       // Criteria for ignoring a node

  void recurseOpenNode(
	FNode* curNode,
	POSVEL_T pos_x,
	POSVEL_T pos_y,
	POSVEL_T pos_z,
	POSVEL_T bhAngle,           // Open node to examine children
	POSVEL_T critRadius,        // Accept or ignore node not opened
	vector<ID_T>* partInteract,
	vector<ID_T>* nodeInteract);

  ////////////////////////////////////////////////////////////////////////////
  //
  // Calculate force on groups particles in a cell
  //
  void treeForceGroup(
	POSVEL_T openAngle,         // Criteria for opening a node
	POSVEL_T critRadius,        // Criteria for ignoring a node
	int minGroup,               // Minimum particles in one group
	int maxGroup);              // Maximum particles in one group

  // Short range force on one particle starting with particle walking up
  void walkTreeGroup(
	ID_T curId,                 // Index of particle or node
	POSVEL_T minMass,           // Group of particles more than this mass
	POSVEL_T maxMass,           // Group of particles less than this mass
	POSVEL_T openAngle,         // Criteria for opening a node
	POSVEL_T critRadius);       // Criteria for ignoring a node

  // Create the interaction list for a particle starting from root
  void createParticleInteractList(
	ID_T p,
	POSVEL_T bhAngle,
	POSVEL_T critRadius,
	vector<ID_T>* partInteract,
	vector<ID_T>* nodeInteract);

  // Create the interaction list for a node starting from root
  void createNodeInteractList(
	ID_T node,
	POSVEL_T bhAngle,
	POSVEL_T critRadius,
	vector<ID_T>* partInteract,
	vector<ID_T>* nodeInteract);

  // Force calculation for a group of particles
  void forceCalculationGroup(
	ID_T node,
	POSVEL_T bhAngle,
	POSVEL_T critRadius,
	vector<ID_T>* partInteract,
	vector<ID_T>* nodeInteract);

  // Like forceCalculation but with extra exclusion test since particles
  // are grouped and not all particles and nodes will apply to each
  POSVEL_T forceCalculationParticle(
	ID_T p0,                    // Index of target particle
	POSVEL_T critRadius,
	vector<ID_T>* partInteract,  // Particles which act on object
	vector<ID_T>* nodeInteract); // Nodes which act on object

  // Collect the particles within the group
  void collectParticles(
	ID_T curId,
	vector<ID_T>* particles);

  ////////////////////////////////////////////////////////////////////////////
  //
  // Force calculations
  //
  POSVEL_T forceCalculation(
	ID_T p0,                    // Index of target particle
	vector<ID_T>* partInteract,  // Particles which act on object
	vector<ID_T>* nodeInteract); // Nodes which act on object

  POSVEL_T forceCalculationFast(
	ID_T p0,                    // Index of target particle
	vector<POSVEL_T>* xInteract,
	vector<POSVEL_T>* yInteract,
	vector<POSVEL_T>* zInteract,
	vector<POSVEL_T>* mInteract);

  // Choose the correct octant for placing a node in the tree
  int getChildIndex(
	FNode* node,
	ID_T pindx);

  // Print BH tree depth first
  void printBHForceTree();

  // Print force values
  void printForceValues();
	
private:
  int    myProc;                // My processor number
  int    numProc;               // Total number of processors

  POSVEL_T boxSize;             // Physical box size of the data set
  POSVEL_T openingAngle;	// Criteria for opening node to lower level

  ID_T   particleCount;         // Total particles
  ID_T   nodeCount;             // Total nodes
  ID_T   nodeOffset;		// Index of first node is after last particle
  POSVEL_T particleMass;	// Average particle mass

  POSVEL_T* xx;                 // X location for particles on this processor
  POSVEL_T* yy;                 // Y location for particles on this processor
  POSVEL_T* zz;                 // Z location for particles on this processor
  POSVEL_T* vx;                 // X velocity for particles on this processor
  POSVEL_T* vy;                 // Y velocity for particles on this processor
  POSVEL_T* vz;                 // Z velocity for particles on this processor
  POSVEL_T* mass;               // Mass for particles on this processor

  POSVEL_T minRange[DIMENSION]; // Physical range of data
  POSVEL_T maxRange[DIMENSION]; // Physical range of data

  vector<FParticle, bigchunk_allocator<FParticle> >  fParticle;	// Leaf particles in tree
  vector<FNode, bigchunk_allocator<FNode> >          fNode;	// Internal nodes of tree

  ForceLaw *m_fl;
  float m_fcoeff;
};

#endif
