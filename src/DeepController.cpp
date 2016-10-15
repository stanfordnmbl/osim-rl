/* ---------------------------------------------------------------- *
 * Adapted from OpenSim ControllerExample.cpp                       *
 * ---------------------------------------------------------------- */

// Include OpenSim and functions
#include <OpenSim/OpenSim.h>
#include "OpenSim/Common/STOFileAdapter.h"

// This allows us to use OpenSim functions, classes, etc., without having to
// prefix the names of those things with "OpenSim::".
using namespace OpenSim;

// This allows us to use SimTK functions, classes, etc., without having to
// prefix the names of those things with "SimTK::".
using namespace SimTK;

//______________________________________________________________________________
/**
 * Run a forward dynamics simulation with a controller attached to a model.
 */
int main()
{
    try {
        // Create an OpenSim model from the model file provided.
      Model osimModel( "models/gait10dof18musc_subject01.osim" );
	osimModel.setUseVisualizer(true);

	// Define the initial and final simulation times.
        double initialTime = 0.0;
        double finalTime = 10.0;

	// Define non-zero (defaults are 0) states for the free joint.
        CoordinateSet& modelCoordinateSet = osimModel.updCoordinateSet();
	
        // Define the initial muscle states.
	const Set<Muscle>& muscleSet = osimModel.getMuscles();

        // Create the controller.
	PrescribedController* brain = new PrescribedController();
	Muscle* hamstrings_r = &muscleSet.get(0);
	brain->addActuator(*hamstrings_r);
	
	// Muscle excitation is 0.3 for the first 0.5 seconds, then increases to 1.
	brain->prescribeControlForActuator("hamstrings_r",
					   new StepFunction(0.5, 3, 0.3, 1));
        osimModel.addController( brain );

        // Initialize the system and get the state representing the
        // system.
        SimTK::State& si = osimModel.initSystem();

        // Compute initial conditions for muscles.
        osimModel.equilibrateMuscles(si);

        // Create the integrator and manager for the simulation.
	    SimTK::RungeKuttaMersonIntegrator
            integrator( osimModel.getMultibodySystem() );
	    integrator.setAccuracy( 1.0e-4 );

        Manager manager( osimModel, integrator );

        // Examine the model.
        osimModel.printDetailedInfo( si, std::cout );

        // Print out the initial position and velocity states.
        for( int i = 0; i < modelCoordinateSet.getSize(); i++ ) {
            std::cout << "Initial " << modelCoordinateSet[i].getName()
                << " = " << modelCoordinateSet[i].getValue( si )
                << ", and speed = "
                << modelCoordinateSet[i].getSpeedValue( si ) << std::endl;
        }

        // Integrate from initial time to final time.
        manager.setInitialTime( initialTime );
        manager.setFinalTime( finalTime );
        std::cout << "\n\nIntegrating from " << initialTime
            << " to " << finalTime << std::endl;
	    manager.integrate( si );

	char c;
	std::cin >> c;
    }
    catch (const std::exception &ex) {
        
        // In case of an exception, print it out to the screen.
        std::cout << ex.what() << std::endl;

        // Return 1 instead of 0 to indicate that something
        // undesirable happened.
        return 1;
    }
    // If this program executed up to this line, return 0 to
    // indicate that the intended lines of code were executed.
    return 0;
}
