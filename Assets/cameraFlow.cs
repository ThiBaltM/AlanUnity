using UnityEngine;

public class cameraFollow : MonoBehaviour
{
    public GameObject Alan;
    public Vector3 decalage;
    private void Start()
    {
    }

    private void LateUpdate()
    {
        transform.position = Alan.transform.position + decalage;
    }

}
