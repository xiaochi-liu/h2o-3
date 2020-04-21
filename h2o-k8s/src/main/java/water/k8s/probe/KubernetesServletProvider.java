package water.k8s.probe;

import water.api.*;
import water.server.ServletMeta;
import water.server.ServletProvider;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class KubernetesServletProvider implements ServletProvider {
    private static final List<ServletMeta> SERVLETS = Collections.unmodifiableList(Arrays.asList(
            new ServletMeta("/3/kubernetes/isLeaderNode", KubernetesProbeServlet.class)
    ));

    @Override
    public List<ServletMeta> servlets() {
        return SERVLETS;
    }

    @Override
    public int priority() {
        return 1;
    }
}
