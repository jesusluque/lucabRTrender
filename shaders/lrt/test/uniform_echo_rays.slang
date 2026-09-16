// Copyright (c) 2026 lucabRTrender contributors.
//
// The same question, asked of a ray generation entry. It traces nothing: a
// pipeline needs a ray generation program and nothing else to be launched,
// and what is under test is the parameter block a launch reads, not the
// traversal. A module of its own because a target without ray tracing cannot
// compile a `raygeneration` entry, and an entry a target cannot compile takes
// its module with it.
module uniform_echo_rays;
import uniform_echo;

[shader("raygeneration")]
void echoRaysGen() {
    echoAt(DispatchRaysIndex().x);
}
