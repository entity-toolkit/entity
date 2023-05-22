set(ENTITY_TEST ${PROJECT_NAME}-test.xc)

add_executable(${ENTITY_TEST} test.cpp)

target_link_libraries(${ENTITY_TEST} PUBLIC ${WRAPPER} ${FRAMEWORK} ${SIMENGINE})
target_include_directories(${ENTITY_TEST} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_include_directories(${ENTITY_TEST} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src/wrapper)
target_include_directories(${ENTITY_TEST} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src/framework)
target_include_directories(${ENTITY_TEST} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src/framework/io)
target_include_directories(${ENTITY_TEST} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src/${engine})

enable_testing()
add_test(${ENTITY_TEST} ${ENTITY_TEST})