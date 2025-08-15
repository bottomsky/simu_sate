using System;
using J2.Propagator;

try
{
    Console.WriteLine("[TestApp] Starting basic end-to-end checks...");

    // J2Orbit roundtrip test
    var elems = new COrbitalElements
    {
        a = 7000e3,
        e = 0.001,
        i = 98 * Math.PI / 180.0,
        O = 40 * Math.PI / 180.0,
        w = 20 * Math.PI / 180.0,
        M = 0.5,
        t = 0
    };

    using var propagator = new J2Orbit(elems);
    var state = propagator.ElementsToState(elems);
    var roundtrip = propagator.StateToElements(state, elems.t);
    Console.WriteLine($"[J2Orbit] Roundtrip: a={roundtrip.a:F3}, e={roundtrip.e:E3}, i={roundtrip.i:E3}");

    // Constellation basic test
    using var constellation = new ConstellationPropagator(0.0);
    constellation.AddSatellite(new CCompactOrbitalElements { a = 7000e3, e = 0.001, i = 98 * Math.PI / 180.0, O = 0, w = 0, M = 0 });
    constellation.AddSatellite(new CCompactOrbitalElements { a = 7000e3, e = 0.001, i = 98 * Math.PI / 180.0, O = 60 * Math.PI / 180.0, w = 0, M = 0 });
    Console.WriteLine($"[Constellation] Count before propagate: {constellation.GetSatelliteCount()}");
    constellation.Propagate(3600.0);
    var st0 = constellation.GetSatelliteState(0);
    Console.WriteLine($"[Constellation] Sat0 pos: [{st0.r[0]:F0}, {st0.r[1]:F0}, {st0.r[2]:F0}]");

    Console.WriteLine("[TestApp] All checks passed.");
}
catch (Exception ex)
{
    Console.WriteLine($"[TestApp] Error: {ex.Message}");
    Environment.Exit(1);
}
