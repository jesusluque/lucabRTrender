lrt_add_module(sched
    SOURCES
        src/FrameClock.cpp
    PUBLIC_LIBS
        lrt::core
        genlock::genlock)
