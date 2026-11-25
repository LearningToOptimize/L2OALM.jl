using Aqua
using ExplicitImports
using JET

@testset "Aqua" begin
    Aqua.test_all(L2OALM)
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(L2OALM) === nothing
    @test check_no_stale_explicit_imports(L2OALM) === nothing
    @test check_all_qualified_accesses_via_owners(L2OALM) === nothing
    @test check_no_self_qualified_accesses(L2OALM) === nothing
    for ext in (:L2OALMLuxExt, :L2OALMBatchNLPKernelsExt)
        mod = Base.get_extension(L2OALM, ext)
        @test mod !== nothing
        @test check_no_implicit_imports(mod) === nothing
        @test check_no_stale_explicit_imports(mod) === nothing
    end
end

@testset "JET" begin
    JET.test_package(L2OALM; target_modules = (L2OALM,))
end
