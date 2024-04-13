import { CButton, CCardTitle, CCol, CContainer, CFormSelect, CRow } from "@coreui/react";
import { CChart } from "@coreui/react-chartjs";
import axios from "axios";
import {
  useCallback,
  useEffect,
  useRef,
  useState
} from "react";
import {
  JsonView,
  collapseAllNested,
  defaultStyles,
} from "react-json-view-lite";
import "./Dashboard.css";
import "react-json-view-lite/dist/index.css";

const API_URL = "http://127.0.0.1:44555/api";

export default function Dashboard() {
  const [selectedExperimentA, setSelectedExperimentA] = useState<string>("");
  const [selectedExperimentB, setSelectedExperimentB] = useState<string>("");
  const [data, setData] = useState({ A: [], B: [] });
  const [experiments, setExperiments] = useState<
    { label: string; value: string }[]
  >([]);
  const selectRefA = useRef(null);
  const selectRefB = useRef(null);

  const fetchExperiments = async () => {
    const results = await axios.get<string[]>(`${API_URL}/experiments`);

    const ex = results.data.map((d) => ({ label: d, value: d }));
    setExperiments(ex);
    setSelectedExperimentA(ex[0].value);
    setSelectedExperimentB(ex[1].value);
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const fetchExperimentData = async () => {
    console.log({ selectedExperimentA, selectedExperimentB });
    const resA = await axios.get(
      `${API_URL}/experiments/${selectedExperimentA}`
    );

    const resB = await axios.get(
      `${API_URL}/experiments/${selectedExperimentB}`
    );
    setData({ A: resA.data, B: resB.data });
  };

  const getAccuracyChartDataset = () => {
    const xpAData = data["A"]
      .sort((a, b) => a.round - b.round)
      .map((x) => x.global_accuracy);
    const xpBData = data["B"]
      .sort((a, b) => a.round - b.round)
      .map((x) => x.global_accuracy);
    const len = (xpAData.length > xpBData.length ? xpAData : xpBData).map(
      (_, i) => i
    );

    return {
      labels: len,
      datasets: [
        {
          label: selectedExperimentA,
          data: xpAData,
        },
        {
          label: selectedExperimentB,
          data: xpBData,
        },
      ],
    };
  };

  const getLossChartDataset = () => {
    const xpAData = data["A"]
      .sort((a, b) => a.round - b.round)
      .map((x) => x.global_loss);
    const xpBData = data["B"]
      .sort((a, b) => a.round - b.round)
      .map((x) => x.global_loss);
    const len = (xpAData.length > xpBData.length ? xpAData : xpBData).map(
      (_, i) => i
    );

    return {
      labels: len,
      datasets: [
        {
          label: selectedExperimentA,
          data: xpAData,
        },
        {
          label: selectedExperimentB,
          data: xpBData,
        },
      ],
    };
  };

  const setExperiment = useCallback((ref, setX) => {
    const selectedValue = ref.current!["value"];
    setX(selectedValue);
  }, []);

  useEffect(() => {
    fetchExperimentData();
  }, [selectedExperimentA, selectedExperimentB]);

  useEffect(() => {
    fetchExperiments();
  }, []);

  return (
    <>
      <div>
        <CContainer>
          <CRow>
            <CCol>Select Experiment A</CCol>
            <CCol xs={8}>
              <CFormSelect
                ref={selectRefA}
                area-label="Select experiment name"
                options={experiments}
              />
            </CCol>
            <CCol>
              <CButton
                color="primary"
                onClick={() =>
                  setExperiment(selectRefA, setSelectedExperimentA)
                }
              >
                Select and load
              </CButton>
            </CCol>
          </CRow>
          <CRow>
            <CCol>Select ExperimentB</CCol>
            <CCol xs={8}>
              <CFormSelect
                ref={selectRefB}
                area-label="Select experiment name"
                options={experiments}
              />
            </CCol>
            <CCol>
              <CButton
                color="primary"
                onClick={() =>
                  setExperiment(selectRefB, setSelectedExperimentB)
                }
              >
                Select and load
              </CButton>
            </CCol>
          </CRow>
        </CContainer>
      </div>
      {data && (
        <div>
          <CContainer>
            <CRow>
              <CCol>
                <CCardTitle>Global accuracy over aggregration rounds</CCardTitle>
                <CChart type="line" data={getAccuracyChartDataset()} />
              </CCol>
              <CCol>
              <CCardTitle>Global loss over aggregration rounds</CCardTitle>
                <CChart type="line" data={getLossChartDataset()} />
              </CCol>
            </CRow>
            <CRow>
              <CCol>
                <JsonView
                  data={data}
                  shouldExpandNode={collapseAllNested}
                  style={defaultStyles}
                />
              </CCol>
            </CRow>
          </CContainer>
        </div>
      )}
    </>
  );
}
