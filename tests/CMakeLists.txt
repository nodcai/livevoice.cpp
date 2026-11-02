# Each test_*.cpp is one executable and one ctest. A test returns 77 when
# what it needs (a model file, an env var) is missing; ctest reports it as
# skipped.

function(lv_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE livevoice)
    target_include_directories(${name} PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/third_party)
    target_compile_definitions(${name} PRIVATE LV_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

lv_add_test(test_smoke)
lv_add_test(test_tokenizer)
lv_add_test(test_qwen2)
lv_add_test(test_diffusion)
lv_add_test(test_decoder)
lv_add_test(test_dfn)
