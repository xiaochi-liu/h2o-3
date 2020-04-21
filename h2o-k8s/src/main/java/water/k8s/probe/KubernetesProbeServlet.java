package water.k8s.probe;

import water.H2O;
import water.util.HttpResponseStatus;
import water.util.Log;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

import static water.k8s.KubernetesEmbeddedConfigProvider.K8S_DESIRED_CLUSTER_SIZE_KEY;

public class KubernetesProbeServlet extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        final int desiredClusterSize = Integer.parseInt(System.getenv(K8S_DESIRED_CLUSTER_SIZE_KEY));
        Log.info(H2O.SELF.isLeaderNode() || H2O.CLOUD.size() < desiredClusterSize);
        if (H2O.SELF == null || H2O.SELF.isLeaderNode() || H2O.CLOUD.size() < desiredClusterSize) {
            resp.setStatus(200);
        } else {
            resp.setStatus(404);
        }
    }
}
