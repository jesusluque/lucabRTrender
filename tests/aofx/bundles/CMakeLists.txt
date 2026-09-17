# Bundles that exist to be loaded, and bundles that exist to be refused
# (openFXplayer's corpus, plus one that runs a kernel with the wrong buffers).
set(LRT_AOFX_TEST_BUNDLES "${CMAKE_BINARY_DIR}/aofx-test-plugins" CACHE INTERNAL "")
set(LRT_AOFX_TEST_RUN_BUNDLES "${CMAKE_BINARY_DIR}/aofx-test-run" CACHE INTERNAL "")

aofx_add_bundle(aofx_test_good NAME good SOURCES good.cpp OUTPUT_DIR ${LRT_AOFX_TEST_BUNDLES})
aofx_add_bundle(aofx_test_wrongabi NAME wrongabi SOURCES wrongabi.cpp OUTPUT_DIR ${LRT_AOFX_TEST_BUNDLES})
aofx_add_bundle(aofx_test_wrongtoolchain NAME wrongtoolchain SOURCES wrongtoolchain.cpp
                OUTPUT_DIR ${LRT_AOFX_TEST_BUNDLES})
aofx_add_bundle(aofx_test_hidden NAME hidden SOURCES hidden.cpp OUTPUT_DIR ${LRT_AOFX_TEST_BUNDLES})
# The one testing what happens without default visibility.
set_target_properties(aofx_test_hidden PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

aofx_add_bundle(aofx_test_reporter NAME reporter SOURCES reporter.cpp OUTPUT_DIR ${LRT_AOFX_TEST_RUN_BUNDLES})

aofx_add_bundle(aofx_test_miscount NAME miscount SOURCES miscount.cpp
                KERNELS miscount ENTRY miscountMain SOURCE miscount.slang
                OUTPUT_DIR ${LRT_AOFX_TEST_RUN_BUNDLES})

add_custom_target(aofx_test_bundles DEPENDS
    aofx_test_good aofx_test_wrongabi aofx_test_wrongtoolchain aofx_test_hidden
    aofx_test_miscount aofx_test_reporter lrt_aofx_invert)
